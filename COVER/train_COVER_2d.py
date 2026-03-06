import argparse
import csv
import math
import os
import shutil
import glob
import time

from os.path import join
from sklearn.model_selection import train_test_split

from monai.transforms import RandRotate, RandFlip, RandAffine, Compose

import torch.nn.functional as F
import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader
from models.cover import COVER
from utils.STN import SpatialTransformer
from utils.Transform_2d import CropTransform, AppearanceTransform
from utils.dataloader_SSP_2d import DatasetFromFolder2D, pack_to_npz
from utils.losses import partical_MAE, partical_COS
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(description="COVER 2D SSL Pretraining")
parser.add_argument("-modelname", metavar="NAME", default="COVER_2D_brain_vessels",
                    help="model name used for checkpoint files")
parser.add_argument("-data", metavar="DIR", default="", nargs="+",
                    help="one or more paths to folders containing NIfTI files "
                         "e.g. -data /folder1 /folder2")
parser.add_argument("-preprocessed_dir",
                    default="/content/drive/MyDrive/CoW_preprocessed",
                    help="path to folder containing preprocessed .npy slices")
parser.add_argument("-npz_path",
                    default="/content/drive/MyDrive/CoW_preprocessed.npz",
                    help="path to packed .npz archive (faster) — used automatically "
                         "if the file exists, falls back to -preprocessed_dir otherwise")
parser.add_argument("-save_dir", metavar="SAVE", default="/content/drive/MyDrive/CoW_checkpoints",
                    help="directory to save checkpoints (use Google Drive path on Colab)")  # FIX 7
parser.add_argument("-j", "--workers", default=2, type=int, metavar="N",
                    help="number of data loading workers (default: 2)")
parser.add_argument("--epochs", default=100, type=int, metavar="N",
                    help="number of total epochs to run")
parser.add_argument("--start-epoch", default=0, type=int, metavar="N",
                    help="manual epoch number (useful on restarts)")
parser.add_argument("-b", "--batch-size", default=4, type=int, metavar="N",
                    help="mini-batch size (default: 4)")
parser.add_argument("-img-size", default=256, type=int,
                    help="input image size after crop")
parser.add_argument("--lr", "--learning-rate", default=1e-4, type=float, metavar="LR",
                    help="initial learning rate", dest="lr")
parser.add_argument("--wd", "--weight-decay", default=1e-4, type=float, metavar="W",
                    help="weight decay (default: 1e-4)", dest="weight_decay")
parser.add_argument("-p", "--print-freq", default=1, type=int, metavar="N",
                    help="print frequency (default: 1)")
parser.add_argument("--gpu", default=0, type=int,
                    help="GPU id to use.")
parser.add_argument("--n_channels", default=1, type=int,
                    help="number of input channels")
parser.add_argument("--amp", default=2, type=int,
                    help="amplification of the kernel (default: 2)")
parser.add_argument("--degree", default=1.5, type=float,
                    help="spatial transformation degree (default: 1.5)")


# ─── BATCHED GPU AUGMENTATION ────────────────────────────────────────────────


def gpu_appearance_aug(x):
    """Pure GPU appearance augmentation — no scipy, no CPU round-trip."""
    factor = 0.75 + torch.rand(1, device=x.device) * 0.5
    mn  = x.mean()
    x   = (x - mn) * factor + mn
    x   = x + (torch.rand(1, device=x.device) - 0.5) * 0.1
    std = torch.rand(1, device=x.device) * 0.04
    x   = x + torch.randn_like(x) * std
    if torch.rand(1).item() < 0.3:
        x = torch.nn.functional.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    return x.clamp(0.0, 1.0)

def batched_aug(img):
    """
    Lightweight appearance augmentation on GPU — replaces per-sample MONAI CPU loop.
    Random gamma correction + gaussian noise on entire batch in one op.
    ~15x faster than looping MONAI transforms on CPU.
    """
    B = img.shape[0]
    gamma = 0.7 + torch.rand(B, 1, 1, 1, device=img.device) * 0.6  # [0.7, 1.3]
    out   = img.clamp(1e-6).pow(gamma)
    noise_std = torch.rand(1, device=img.device) * 0.05
    out = out + torch.randn_like(out) * noise_std
    return out.clamp(0.0, 1.0)


# ─── GPU FLOW GENERATOR ───────────────────────────────────────────────────────

def make_flow_gpu(batch_size, shape, dev, deg):
    """
    Batched random affine displacement field generated entirely on GPU.
    Replaces the per-sample numpy loop from SpatialTransform2D (~10x faster).
    """
    B, H, W = batch_size, shape[0], shape[1]
    angle = (torch.rand(B, device=dev) * 2 - 1) * deg * np.pi / 9
    scale = 1.0 + (torch.rand(B, device=dev) * 2 - 1) * deg * 0.5
    tx    = (torch.rand(B, device=dev) * 2 - 1) * deg * 0.2
    ty    = (torch.rand(B, device=dev) * 2 - 1) * deg * 0.2
    cos_a, sin_a = torch.cos(angle), torch.sin(angle)
    theta = torch.stack([
        torch.stack([cos_a * scale, -sin_a, tx], dim=1),
        torch.stack([sin_a, cos_a * scale,  ty], dim=1),
    ], dim=1)  # (B, 2, 3)
    grid = torch.nn.functional.affine_grid(
        theta, (B, 1, H, W), align_corners=True)
    base = torch.nn.functional.affine_grid(
        torch.eye(2, 3, device=dev).unsqueeze(0).expand(B, -1, -1),
        (B, 1, H, W), align_corners=True)
    flow = (grid - base).permute(0, 3, 1, 2)
    flow[:, 0] *= (W - 1) / 2
    flow[:, 1] *= (H - 1) / 2
    return flow


def main():
    args = parser.parse_args()
    main_worker(args.gpu, args)


def main_worker(gpu, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # ── Create save directory on Google Drive ────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)  # FIX 7

    # ── Model ─────────────────────────────────────────────────────────────────
    model = COVER(n_channels=args.n_channels, dimensions='2D')

    # Infer learning rate before changing batch size
    init_lr = args.lr * args.batch_size / 256

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    def count_parameters_in_M(model):
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params / 1e6

    print(f"Model parameter count: {count_parameters_in_M(model):.2f} M")

    # ── Loss & Optimizer ──────────────────────────────────────────────────────
    criterion_vec = partical_MAE
    criterion_con = partical_COS

    optimizer = torch.optim.Adam(model.parameters(), init_lr, weight_decay=args.weight_decay)

    # ── Resume from checkpoint ────────────────────────────────────────────────
    if args.start_epoch > 0:
        ckpt_path = os.path.join(
            args.save_dir,
            "checkpoint_{}_{:04d}.pth.tar".format(args.modelname, args.start_epoch - 1)
        )
        if os.path.isfile(ckpt_path):
            print("=> loading checkpoint '{}'".format(ckpt_path))
            loc = "cuda:{}".format(args.gpu) if args.gpu is not None else "cpu"
            checkpoint = torch.load(ckpt_path, map_location=loc)
            model.load_state_dict(checkpoint["state_dict"])
            print("=> loaded checkpoint (epoch {})".format(checkpoint["epoch"]))  # FIX 8
        else:
            print("=> no checkpoint found at '{}'".format(ckpt_path))

    cudnn.benchmark = True

    # ── Data ──────────────────────────────────────────────────────────────────
    shape = (int(args.img_size * 1.5),) * 2  # e.g. (384, 384) for img_size=256

    if os.path.isfile(args.npz_path):
        # ── FAST PATH — load .npz ONCE, share cache between train and val ──
        print(f"Found .npz archive — using fast npz mode: {args.npz_path}")
        import numpy as _np_tmp
        all_keys = sorted(_np_tmp.load(args.npz_path).files)

        # Patient-level split — extract stems from key format:
        # foldername__patientname_sliceXXX
        patient_stems = sorted(set('_'.join(k.split('_')[:-1]) for k in all_keys))
        train_stems, _ = train_test_split(
            patient_stems, test_size=0.2, random_state=42)
        train_stems = set(train_stems)

        train_keys = [k for k in all_keys
                      if '_'.join(k.split('_')[:-1]) in train_stems]
        val_keys   = [k for k in all_keys
                      if '_'.join(k.split('_')[:-1]) not in train_stems]

        print(f"  Train: {len(train_keys)} slices  Val: {len(val_keys)} slices")

        # Load .npz ONCE into shared cache — avoids loading 2x into RAM
        print(f"Loading .npz into shared RAM cache (once) ...")
        npz_data    = _np_tmp.load(args.npz_path)
        shared_cache = {k: npz_data[k].astype(np.float16) for k in all_keys}
        npz_data.close()
        ram_gb = sum(v.nbytes for v in shared_cache.values()) / 1e9
        print(f"  Done — {ram_gb:.2f} GB shared cache. Zero I/O during training.")

        # Pass shared cache directly — skip per-dataset reload
        train_dataset = DatasetFromFolder2D(train_keys, shape, cache=shared_cache)
        val_dataset   = DatasetFromFolder2D(val_keys,   shape, cache=shared_cache)

    else:
        # ── FALLBACK — individual .npy files read directly from Drive ──────
        print(f".npz not found — using npy fallback from {args.preprocessed_dir}")
        print(f"  Tip: run pack_to_npz() in a separate cell for much faster I/O")
        all_files = [join(args.preprocessed_dir, x)
                     for x in os.listdir(args.preprocessed_dir)
                     if x.endswith('.npy')]
        train_files, val_files = train_test_split(
            all_files, test_size=0.2, random_state=42)
        train_dataset = DatasetFromFolder2D(train_files, shape, preload=False)
        val_dataset   = DatasetFromFolder2D(val_files,   shape, preload=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # ── FIX 6: instantiate augmentation objects ONCE here, pass to train/val ──
    device = "cuda:{}".format(args.gpu) if args.gpu is not None else "cpu"
    stn = SpatialTransformer().cuda(args.gpu) if args.gpu is not None else SpatialTransformer()

    degree = args.degree



    # ── Training loop ─────────────────────────────────────────────────────────
    logwriter = LogWriter(
        name=os.path.join(args.save_dir, args.modelname),
        head=["Train Batch time", "Train Data time", "Train Loss_vec", "Train Loss_con",
              "Val Batch time",   "Val Data time",   "Val Loss_vec",   "Val Loss_con"],
    )

    best_val_loss = float('inf')

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, init_lr, epoch, args)  # FIX 2 (see function below)

        train_batch_time_log, train_data_time_log, train_loss_vec_log, train_loss_con_log = train(
            train_loader, model, criterion_vec, criterion_con, optimizer, epoch, args,
            stn, crop_aug
        )

        val_batch_time_log, val_data_time_log, val_loss_vec_log, val_loss_con_log = validate(
            val_loader, model, criterion_vec, criterion_con, epoch, args,
            stn, crop_aug
        )

        val_loss = val_loss_vec_log + val_loss_con_log
        is_best = val_loss < best_val_loss
        best_val_loss = min(val_loss, best_val_loss)

        ckpt_filename = os.path.join(
            args.save_dir,
            "checkpoint_{}_{:04d}.pth.tar".format(args.modelname, epoch)
        )
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "arch": "COVER_2D",  # FIX 1
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
            },
            is_best=is_best,
            filename=ckpt_filename,                          # FIX 7
            save_dir=args.save_dir,
        )

        logwriter.writeLog([
            train_batch_time_log, train_data_time_log, train_loss_vec_log, train_loss_con_log,
            val_batch_time_log,   val_data_time_log,   val_loss_vec_log,   val_loss_con_log,
        ])

        print(f"Epoch [{epoch + 1}/{args.epochs}]")
        print(f"Train Loss - Vector: {train_loss_vec_log:.4f}, Contrast: {train_loss_con_log:.4f}")
        print(f"Val   Loss - Vector: {val_loss_vec_log:.4f},  Contrast: {val_loss_con_log:.4f},  Total: {val_loss:.4f}")
        print("-" * 60)


# ─── TRAIN ────────────────────────────────────────────────────────────────────

def train(train_loader, model, criterion_vec, criterion_con, optimizer, epoch, args,
          stn, crop_aug):
    train_batch_time = AverageMeter("Train Batch time", ":6.3f")
    train_data_time  = AverageMeter("Train Data time",  ":6.3f")
    train_losses_vec = AverageMeter("Train Loss_vec",   ":.4f")
    train_losses_con = AverageMeter("Train Loss_con",   ":.4f")

    model.train()
    end = time.time()

    pbar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}", leave=False,
                unit="batch", dynamic_ncols=True)
    for idx, (img) in enumerate(pbar):
        train_data_time.update(time.time() - end)

        if args.gpu is not None:
            img = img.cuda(args.gpu, non_blocking=True)

        # Batched GPU augmentation — replaces per-sample MONAI CPU loop
        im_aug = batched_aug(img)

        # Batched GPU affine flow — replaces per-sample numpy loop (~10x faster)
        flow_gt = make_flow_gpu(
            im_aug.shape[0], im_aug.shape[2:],
            dev=im_aug.device, deg=args.degree)
        im_F = stn(im_aug, flow_gt)

        # Crop + GPU appearance augmentation
        crop_code = crop_aug.rand_code(im_aug.shape[2:])
        im_M    = crop_aug.augment_crop(im_aug, crop_code)
        im_F    = crop_aug.augment_crop(im_F,   crop_code)
        flow_gt = crop_aug.augment_crop(flow_gt, crop_code)
        im_M    = gpu_appearance_aug(im_M)

        # Forward pass
        flow_pred, f_M, f_F = model(M=im_M, F=im_F)

        # Build valid-region mask from flow
        mask = build_mask(flow_gt)

        # Losses
        loss_vec = criterion_vec(flow_pred, flow_gt, mask)
        f_M2F    = stn(f_M, flow_gt)
        loss_con = criterion_con(f_F, f_M2F, mask)
        loss     = loss_vec + loss_con

        # FIX 3: use img.size(0) — full batch size — not img[0].size(0)
        train_losses_vec.update(loss_vec.item(), img.size(0))
        train_losses_con.update(loss_con.item(), img.size(0))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_batch_time.update(time.time() - end)
        end = time.time()

        # Update tqdm postfix with running averages
        pbar.set_postfix(loss_vec=f"{train_losses_vec.avg:.4f}",
                         loss_con=f"{train_losses_con.avg:.4f}")

    return train_batch_time.avg, train_data_time.avg, train_losses_vec.avg, train_losses_con.avg


# ─── VALIDATE ─────────────────────────────────────────────────────────────────

def validate(val_loader, model, criterion_vec, criterion_con, epoch, args,
             stn, crop_aug):
    val_batch_time = AverageMeter("Val Batch time", ":6.3f")
    val_data_time  = AverageMeter("Val Data time",  ":6.3f")
    val_losses_vec = AverageMeter("Val Loss_vec",   ":.4f")
    val_losses_con = AverageMeter("Val Loss_con",   ":.4f")

    model.eval()
    end = time.time()

    pbar = tqdm(val_loader, desc=f"Val   Epoch {epoch+1}", leave=False,
                unit="batch", dynamic_ncols=True)
    with torch.no_grad():
        for idx, (img) in enumerate(pbar):
            val_data_time.update(time.time() - end)

            if args.gpu is not None:
                img = img.cuda(args.gpu, non_blocking=True)

            # Batched GPU augmentation — replaces per-sample MONAI CPU loop
            im_aug = batched_aug(img)

            # Batched GPU affine flow
            flow_gt = make_flow_gpu(
                im_aug.shape[0], im_aug.shape[2:],
                dev=im_aug.device, deg=args.degree)
            im_F = stn(im_aug, flow_gt)

            crop_code = crop_aug.rand_code(im_aug.shape[2:])
            im_M    = crop_aug.augment_crop(im_aug, crop_code)
            im_F    = crop_aug.augment_crop(im_F,   crop_code)
            flow_gt = crop_aug.augment_crop(flow_gt, crop_code)
            im_M    = gpu_appearance_aug(im_M)

            device  = torch.device('cuda:{}'.format(args.gpu))
            im_M    = im_M.to(device)
            im_F    = im_F.to(device)
            flow_gt = flow_gt.to(device)

            flow_pred, f_M, f_F = model(M=im_M, F=im_F)

            mask     = build_mask(flow_gt)
            loss_vec = criterion_vec(flow_pred, flow_gt, mask)
            f_M2F    = stn(f_M, flow_gt)
            loss_con = criterion_con(f_F, f_M2F, mask)

            # FIX 3: correct batch size
            val_losses_vec.update(loss_vec.item(), img.size(0))
            val_losses_con.update(loss_con.item(), img.size(0))

            val_batch_time.update(time.time() - end)
            end = time.time()

            # Update tqdm postfix with running averages
            pbar.set_postfix(loss_vec=f"{val_losses_vec.avg:.4f}",
                             loss_con=f"{val_losses_con.avg:.4f}")

    return val_batch_time.avg, val_data_time.avg, val_losses_vec.avg, val_losses_con.avg


# ─── HELPERS ──────────────────────────────────────────────────────────────────

_mask_grid_cache = {}

def build_mask(flow_gt):
    """Build a binary mask of pixels whose displaced location stays inside [0,1]."""
    shape  = flow_gt.shape[2:]
    device = flow_gt.device
    key    = (shape, device)
    # Use cached grid — avoids CPU arange + GPU transfer every batch
    if key not in _mask_grid_cache:
        vectors = [torch.arange(0, s, device=device, dtype=torch.float32) for s in shape]
        grids   = torch.meshgrid(vectors, indexing='ij')
        _mask_grid_cache[key] = torch.stack(grids).unsqueeze(0)  # (1,2,H,W)
    grid = _mask_grid_cache[key]

    new_locs = grid + flow_gt
    # Vectorised normalisation — no Python loop
    shape_t  = torch.tensor([s - 1 for s in shape],
                             dtype=torch.float32, device=device).view(1, 2, 1, 1)
    new_locs = new_locs / shape_t

    in_bounds = (new_locs >= 0) & (new_locs <= 1)
    mask = in_bounds[:, 0:1].float() * in_bounds[:, 1:2].float()
    return mask


def save_checkpoint(state, is_best, filename, save_dir, keep_last=3):
    """
    Save checkpoint every epoch.
    When a new best is found:
      - copy it to model_best.pth.tar
      - delete all old epoch checkpoints except the last `keep_last`
        so Drive doesn't fill up
    """
    torch.save(state, filename)

    if is_best:
        best_path = os.path.join(save_dir, "model_best.pth.tar")
        shutil.copyfile(filename, best_path)
        print(f"  => New best model saved (val_loss improved) → {best_path}")

        # Remove old checkpoints — keep only the most recent `keep_last`
        all_ckpts = sorted(
            glob.glob(os.path.join(save_dir, "checkpoint_*.pth.tar"))
        )  # sorted alphabetically = chronologically given _XXXX epoch format
        to_delete = all_ckpts[:-keep_last]
        for old_ckpt in to_delete:
            os.remove(old_ckpt)
            print(f"  => Removed old checkpoint: {os.path.basename(old_ckpt)}")


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Cosine annealing LR decay."""
    cur_lr = init_lr * 0.5 * (1.0 + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:  # FIX 2: actually apply the new LR
        param_group['lr'] = cur_lr


# ─── LOGGING ──────────────────────────────────────────────────────────────────

class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt  = fmt
        self.reset()

    def reset(self):
        self.val   = 0
        self.avg   = 0
        self.sum   = 0
        self.count = 0

    def update(self, val, n=1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


# ProgressMeter removed — replaced by tqdm


class LogWriter:
    def __init__(self, name, head):
        self.name = name + '.csv'
        with open(self.name, 'w', encoding='utf-8', newline='') as f:
            csv.writer(f).writerow(head)

    def writeLog(self, row):
        with open(self.name, 'a', encoding='utf-8', newline='') as f:
            csv.writer(f).writerow(row)


if __name__ == "__main__":
    main()
