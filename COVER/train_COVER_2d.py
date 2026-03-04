import argparse
import csv
import math
import os
import shutil
import glob
import time

import nibabel as nib
from os.path import join
from os import listdir
from sklearn.model_selection import train_test_split

from monai.transforms import RandRotate, RandFlip, RandAffine, Compose

import torch.nn.functional as F
import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader
from models.cover import COVER
from utils.STN import SpatialTransformer
from utils.Transform_2d import SpatialTransform2D, CropTransform, AppearanceTransform
from utils.dataloader_SSP_2d import DatasetFromFolder2D, preprocess_to_npy, copy_to_ramdisk
from utils.losses import partical_MAE, partical_COS
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(description="COVER 2D SSL Pretraining")
parser.add_argument("-modelname", metavar="NAME", default="COVER_2D_brain_vessels",
                    help="model name used for checkpoint files")
parser.add_argument("-data", metavar="DIR", default="", nargs="+",
                    help="one or more paths to folders containing NIfTI files "
                         "e.g. -data /folder1 /folder2")
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
    if not os.path.exists('data/preprocessed') or len(os.listdir('data/preprocessed')) == 0:
        # args.data is a list of folders — works for single or multiple
        preprocess_to_npy(args.data, 'data/preprocessed')
        print("Data preprocessing completed and saved to 'data/preprocessed'.")

    # Copy preprocessed slices to /dev/shm RAM disk for faster I/O
    fast_dir  = copy_to_ramdisk('data/preprocessed')
    all_files = [join(fast_dir, x)
                 for x in os.listdir(fast_dir) if x.endswith('.npy')]
    train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)

    # preload=True loads all slices into RAM dict — zero disk I/O per epoch
    train_dataset = DatasetFromFolder2D(train_files, (int(args.img_size * 1.5),) * 2, preload=True)
    val_dataset   = DatasetFromFolder2D(val_files,   (int(args.img_size * 1.5),) * 2, preload=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # ── FIX 6: instantiate augmentation objects ONCE here, pass to train/val ──
    device = "cuda:{}".format(args.gpu) if args.gpu is not None else "cpu"
    stn = SpatialTransformer().cuda(args.gpu) if args.gpu is not None else SpatialTransformer()
    app_aug = AppearanceTransform()
    crop_aug = CropTransform((args.img_size, args.img_size))

    degree = args.degree
    spatial_aug = SpatialTransform2D(
        do_rotation=True,
        angle_x=(degree * -np.pi / 9, degree * np.pi / 9),
        angle_y=(degree * -np.pi / 9, degree * np.pi / 9),
        do_scale=True,
        scale_x=(1 - degree * 0.5, 1 + degree * 0.5),
        scale_y=(1 - degree * 0.5, 1 + degree * 0.5),
        do_translate=True,
        trans_x=(degree * -0.2, degree * 0.2),
        trans_y=(degree * -0.2, degree * 0.2),
        do_shear=True,
        shear_xy=(degree * -np.pi / 32, degree * np.pi / 32),
        shear_yx=(degree * -np.pi / 32, degree * np.pi / 32),
        do_elastic_deform=False,
        device=device,
    )

    monai_aug = Compose([
        RandRotate(range_x=0.1, prob=0.4),
        RandFlip(spatial_axis=0, prob=0.5),
        RandAffine(
            prob=0.3,
            rotate_range=(0.1, 0.1),
            scale_range=(0.1, 0.1),
            translate_range=(10, 10),
            padding_mode="zeros",
        ),
    ])

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
            stn, app_aug, spatial_aug, crop_aug, monai_aug  # FIX 6
        )

        val_batch_time_log, val_data_time_log, val_loss_vec_log, val_loss_con_log = validate(
            val_loader, model, criterion_vec, criterion_con, epoch, args,
            stn, app_aug, spatial_aug, crop_aug, monai_aug  # FIX 6
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
          stn, app_aug, spatial_aug, crop_aug, monai_aug):  # FIX 6: received as args
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

        # FIX 5: apply MONAI transforms per sample (expects (C,H,W), not (B,C,H,W))
        # MONAI moves tensors to CPU internally — explicit .cpu() then re-send to GPU
        im_aug = torch.stack([monai_aug(img[i].cpu()) for i in range(img.shape[0])])
        im_aug = im_aug.cuda(args.gpu, non_blocking=True)  # ✅ back to GPU

        # Spatial transformation → deformed image + ground-truth flow
        flow_gt = []
        for j in range(im_aug.shape[0]):
            flow, _ = spatial_aug.rand_coords(im_aug.shape[2:])
            flow_gt.append(flow)
        flow_gt = torch.cat(flow_gt, dim=0)
        im_F = stn(im_aug, flow_gt)

        # Crop + appearance augmentation
        crop_code = crop_aug.rand_code(im_aug.shape[2:])
        im_M      = crop_aug.augment_crop(im_aug, crop_code)
        im_M      = app_aug.rand_aug(im_M)
        im_F      = crop_aug.augment_crop(im_F, crop_code)
        flow_gt   = crop_aug.augment_crop(flow_gt, crop_code)

        # crop_aug / app_aug may return CPU tensors — re-send to GPU
        device  = torch.device('cuda:{}'.format(args.gpu))
        im_M    = im_M.to(device)
        im_F    = im_F.to(device)
        flow_gt = flow_gt.to(device)

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
             stn, app_aug, spatial_aug, crop_aug, monai_aug):  # FIX 6
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

            # FIX 5: per-sample MONAI augmentation — explicit .cpu() then back to GPU
            im_aug = torch.stack([monai_aug(img[i].cpu()) for i in range(img.shape[0])])
            im_aug = im_aug.cuda(args.gpu, non_blocking=True)  # ✅ back to GPU

            flow_gt = []
            for j in range(im_aug.shape[0]):
                flow, _ = spatial_aug.rand_coords(im_aug.shape[2:])
                flow_gt.append(flow)
            flow_gt = torch.cat(flow_gt, dim=0)
            im_F = stn(im_aug, flow_gt)

            crop_code = crop_aug.rand_code(im_aug.shape[2:])
            im_M      = crop_aug.augment_crop(im_aug, crop_code)
            im_M      = app_aug.rand_aug(im_M)
            im_F      = crop_aug.augment_crop(im_F, crop_code)
            flow_gt   = crop_aug.augment_crop(flow_gt, crop_code)

            # crop_aug / app_aug may return CPU tensors — re-send to GPU
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

def build_mask(flow_gt):
    """Build a binary mask of pixels whose displaced location stays inside [0,1]."""
    shape = flow_gt.shape[2:]
    vectors = [torch.arange(0, s) for s in shape]
    # FIX 4: explicit indexing='ij' to silence deprecation warning
    grids = torch.meshgrid(vectors, indexing='ij')
    grid  = torch.stack(grids)
    grid  = torch.unsqueeze(grid, 0).float().to(flow_gt.device)

    new_locs = grid + flow_gt
    for i in range(len(shape)):
        new_locs[:, i, ...] = new_locs[:, i, ...] / (shape[i] - 1)

    in_bounds = (new_locs >= 0) & (new_locs <= 1)               # bool tensor
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
