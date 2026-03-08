"""
Attention U-Net 2D — Circle of Willis Segmentation
====================================================
- Input  : NIfTI CTA volumes (imagesTr/) + binary masks (labelsTr/)
- Model  : MONAI AttentionUnet 2D, optionally initialised from COVER SSL checkpoint
- Loss   : DiceCELoss + soft clDice
- Metrics: Dice + Hausdorff95
- Output : best checkpoint + CSV loss log + kagglehub backup

Folder structure expected:
    <image_dir>/topcow_ct_XXX_0000.nii.gz
    <mask_dir> /topcow_ct_XXX.nii.gz

Pairing: strip '_0000' from image stem → mask stem
"""

import argparse
import csv
import gc
import glob
import os
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import monai
from monai.data import DataLoader, Dataset
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.networks.nets import AttentionUnet
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    Spacingd,
    ToTensord,
)
from monai.utils import set_determinism

# Optional kagglehub backup
try:
    import kagglehub
    KAGGLEHUB_AVAILABLE = True
except ImportError:
    KAGGLEHUB_AVAILABLE = False

set_determinism(seed=42)

# ─── ARGUMENT PARSER ──────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser(description="Attention U-Net 2D — CoW Segmentation")

    parser.add_argument("-image_dir",  default="/content/drive/MyDrive/imagesTr",
                        help="folder containing CTA images (*_0000.nii.gz)")
    parser.add_argument("-mask_dir",   default="/content/drive/MyDrive/labelsTr",
                        help="folder containing segmentation masks (*.nii.gz)")
    parser.add_argument("-save_dir",   default="/kaggle/working/seg_checkpoints",
                        help="where to save checkpoints and logs")
    parser.add_argument("-npz_path",   default="",
                        help="path to preprocessed .npz (if already done)")
    parser.add_argument("-ssl_checkpoint", default="",
                        help="path to COVER SSL checkpoint for encoder init")
    parser.add_argument("-modelname",  default="AttUNet_CoW")
    parser.add_argument("-kaggle_dataset", default="",
                        help="Kaggle dataset handle for backup e.g. 'user/cow-seg-checkpoints'")
    parser.add_argument("--backup_every", default=5,  type=int)
    parser.add_argument("--epochs",       default=70, type=int)
    parser.add_argument("--batch_size",   default=32,  type=int)
    parser.add_argument("--lr",           default=1e-4, type=float)
    parser.add_argument("--img_size",     default=256,  type=int)
    parser.add_argument("--val_interval", default=1,    type=int)
    parser.add_argument("--freeze_epochs",default=10,    type=int,
                        help="freeze encoder for first N epochs (phase 1)")
    parser.add_argument("--workers",      default=0,    type=int)
    parser.add_argument("--gpu",          default=0,    type=int)
    return parser.parse_args()

# ─── SOFT CLDICE LOSS ─────────────────────────────────────────────────────────

class SoftClDiceLoss(nn.Module):
    """
    Soft clDice loss — differentiable approximation of topology-preserving
    skeleton overlap. Critical for thin vessel structures like CoW.
    Reference: Shit et al., CVPR 2021
    """
    def __init__(self, iter_=3, smooth=1.0):
        super().__init__()
        self.iter   = iter_
        self.smooth = smooth

    def soft_erode(self, img):
        p1 = -F.max_pool2d(-img, kernel_size=(3,1), stride=(1,1), padding=(1,0))
        p2 = -F.max_pool2d(-img, kernel_size=(1,3), stride=(1,1), padding=(0,1))
        return torch.min(p1, p2)

    def soft_dilate(self, img):
        return F.max_pool2d(img, kernel_size=(3,3), stride=(1,1), padding=(1,1))

    def soft_open(self, img):
        return self.soft_dilate(self.soft_erode(img))

    def soft_skel(self, img):
        img1  = self.soft_open(img)
        skel  = F.relu(img - img1)
        for _ in range(self.iter - 1):
            img   = self.soft_erode(img)
            img1  = self.soft_open(img)
            delta = F.relu(img - img1)
            skel  = skel + F.relu(delta - skel * delta)
        return skel

    def forward(self, pred, target):
        pred   = torch.sigmoid(pred)
        skel_p = self.soft_skel(pred)
        skel_t = self.soft_skel(target)
        tprec  = (torch.sum(skel_p * target) + self.smooth) / (torch.sum(skel_p) + self.smooth)
        tsens  = (torch.sum(skel_t * pred)   + self.smooth) / (torch.sum(skel_t) + self.smooth)
        cl_dice = 1.0 - 2.0 * tprec * tsens / (tprec + tsens)
        return cl_dice

# ─── COMBINED LOSS ────────────────────────────────────────────────────────────

class CombinedLoss(nn.Module):
    def __init__(self, cldice_weight=0.3, cldice_start_epoch=20):
        super().__init__()
        # Phase A (epochs 1-20):   DiceLoss only — stable learning from scratch
        # Phase B (epochs 21-120): 70% Dice + 30% clDice — topology refinement
        # clDice is unstable when predictions are random noise (early training)
        self.dice              = monai.losses.DiceLoss(sigmoid=True)
        self.cldice            = SoftClDiceLoss()
        self.cldice_weight     = cldice_weight
        self.cldice_start_epoch = cldice_start_epoch
        self.current_epoch     = 0  # updated externally each epoch

    def forward(self, pred, target):
        loss_dice = self.dice(pred, target)
        if self.current_epoch >= self.cldice_start_epoch:
            loss_cl = self.cldice(pred, target)
            return (1 - self.cldice_weight) * loss_dice + self.cldice_weight * loss_cl
        return loss_dice

# ─── PREPROCESSING ────────────────────────────────────────────────────────────

def preprocess_volumes_to_npz(image_dir, mask_dir, npz_path,
                               img_size=256, hu_min=100, hu_max=500):
    """
    Load NIfTI volumes, extract 2D axial slices, save as single .npz.
    Pairs images/masks by stripping '_0000' from image stem.
    Skips blank slices (no vessel voxels).
    """
    import nibabel as nib
    import cv2

    # Build pairs
    image_files = sorted(glob.glob(os.path.join(image_dir, "*_0000.nii*")))
    pairs = []
    for img_path in image_files:
        basename = os.path.basename(img_path)
        stem     = basename.replace("_0000.nii.gz", "").replace("_0000.nii", "")
        msk_path = os.path.join(mask_dir, stem + ".nii")
        if not os.path.isfile(msk_path):
            msk_path = os.path.join(mask_dir, stem + ".nii.gz")
        if os.path.isfile(msk_path):
            pairs.append((img_path, msk_path, stem))
        else:
            print(f"  ⚠️  No mask found for {stem} — skipping")

    print(f"Found {len(pairs)} image/mask pairs")

    images_dict = {}
    masks_dict  = {}
    slice_count = 0
    skipped     = 0

    for img_path, msk_path, stem in tqdm(pairs, desc="Preprocessing", mininterval=10):
        img_vol = nib.load(img_path).get_fdata().astype(np.float32)
        msk_vol = nib.load(msk_path).get_fdata().astype(np.float32)

        # Binary mask
        msk_vol = (msk_vol > 0).astype(np.float32)

        for s in range(img_vol.shape[2]):
            img_sl = img_vol[:, :, s]
            msk_sl = msk_vol[:, :, s]

            # Skip truly blank image slices only (no signal at all)
            if img_sl.max() == 0:
                skipped += 1
                continue

            # HU windowing + normalise
            img_sl = np.clip(img_sl, hu_min, hu_max)
            img_sl = (img_sl - hu_min) / (hu_max - hu_min)

            # Resize
            img_sl = cv2.resize(img_sl, (img_size, img_size),
                                interpolation=cv2.INTER_LINEAR)
            msk_sl = cv2.resize(msk_sl, (img_size, img_size),
                                interpolation=cv2.INTER_NEAREST)

            key = f"{stem}_slice{s:03d}"
            images_dict[key] = img_sl.astype(np.float16)
            masks_dict[key]  = msk_sl.astype(np.float16)
            slice_count += 1

    print(f"Saving {slice_count} slices to {npz_path} ...")
    np.savez_compressed(npz_path,
                        **{f"img_{k}": v for k, v in images_dict.items()},
                        **{f"msk_{k}": v for k, v in masks_dict.items()})
    size_gb = os.path.getsize(npz_path) / 1e9
    print(f"Done — {npz_path} ({size_gb:.2f} GB), {skipped} blank slices skipped")

# ─── DATASET ──────────────────────────────────────────────────────────────────

class SliceDataset(torch.utils.data.Dataset):
    """
    Loads 2D image/mask slice pairs from a shared .npz cache.
    Applies augmentation on GPU if augment=True.
    """
    def __init__(self, keys, cache, img_size=256, augment=False):
        self.keys     = keys
        self.cache    = cache
        self.img_size = img_size
        self.augment  = augment

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        img = torch.from_numpy(
            self.cache[f"img_{key}"].astype(np.float32)).unsqueeze(0)  # (1,H,W)
        msk = torch.from_numpy(
            self.cache[f"msk_{key}"].astype(np.float32)).unsqueeze(0)  # (1,H,W)
        return img, msk

# ─── GPU AUGMENTATION ─────────────────────────────────────────────────────────

def gpu_augment(img, msk, device):
    """Batched GPU augmentation for image+mask pairs."""
    B = img.shape[0]

    # Random horizontal flip
    if torch.rand(1).item() < 0.5:
        img = torch.flip(img, dims=[-1])
        msk = torch.flip(msk, dims=[-1])

    # Random vertical flip
    if torch.rand(1).item() < 0.5:
        img = torch.flip(img, dims=[-2])
        msk = torch.flip(msk, dims=[-2])

    # Random affine (rotation + scale) — applied identically to img and mask
    angle = (torch.rand(B, device=device) * 2 - 1) * 0.2  # ±~11 degrees
    scale = 1.0 + (torch.rand(B, device=device) * 2 - 1) * 0.1
    cos_a, sin_a = torch.cos(angle), torch.sin(angle)
    theta = torch.stack([
        torch.stack([cos_a * scale, -sin_a,
                     torch.zeros(B, device=device)], dim=1),
        torch.stack([sin_a, cos_a * scale,
                     torch.zeros(B, device=device)], dim=1),
    ], dim=1)
    grid = F.affine_grid(theta, img.shape, align_corners=True)
    img  = F.grid_sample(img, grid, mode='bilinear',
                         padding_mode='zeros', align_corners=True)
    msk  = F.grid_sample(msk, grid, mode='nearest',
                         padding_mode='zeros', align_corners=True)

    # Appearance augmentation on image only
    gamma = 0.7 + torch.rand(B, 1, 1, 1, device=device) * 0.6
    img   = img.clamp(1e-6).pow(gamma)
    noise = torch.randn_like(img) * torch.rand(1, device=device) * 0.04
    img   = (img + noise).clamp(0.0, 1.0)

    return img, msk

# ─── SSL ENCODER TRANSFER ─────────────────────────────────────────────────────

# Explicit layer mapping: COVER backbone → MONAI AttentionUnet encoder
# Derived from printing both model's named_parameters()
COVER_TO_ATTUNET = {
    # inc → model.0 (first encoder block)
    "backbone.inc.double_conv.0.weight":  "model.0.conv.0.conv.weight",
    "backbone.inc.double_conv.0.bias":    "model.0.conv.0.conv.bias",
    "backbone.inc.double_conv.1.weight":  "model.0.conv.0.adn.N.weight",
    "backbone.inc.double_conv.1.bias":    "model.0.conv.0.adn.N.bias",
    "backbone.inc.double_conv.3.weight":  "model.0.conv.1.conv.weight",
    "backbone.inc.double_conv.3.bias":    "model.0.conv.1.conv.bias",
    "backbone.inc.double_conv.4.weight":  "model.0.conv.1.adn.N.weight",
    "backbone.inc.double_conv.4.bias":    "model.0.conv.1.adn.N.bias",

    # down1 → model.1.submodule.0 (encoder block 2)
    "backbone.down1.maxpool_conv.1.double_conv.0.weight": "model.1.submodule.0.conv.0.conv.weight",
    "backbone.down1.maxpool_conv.1.double_conv.0.bias":   "model.1.submodule.0.conv.0.conv.bias",
    "backbone.down1.maxpool_conv.1.double_conv.1.weight": "model.1.submodule.0.conv.0.adn.N.weight",
    "backbone.down1.maxpool_conv.1.double_conv.1.bias":   "model.1.submodule.0.conv.0.adn.N.bias",
    "backbone.down1.maxpool_conv.1.double_conv.3.weight": "model.1.submodule.0.conv.1.conv.weight",
    "backbone.down1.maxpool_conv.1.double_conv.3.bias":   "model.1.submodule.0.conv.1.conv.bias",
    "backbone.down1.maxpool_conv.1.double_conv.4.weight": "model.1.submodule.0.conv.1.adn.N.weight",
    "backbone.down1.maxpool_conv.1.double_conv.4.bias":   "model.1.submodule.0.conv.1.adn.N.bias",

    # down2 → model.1.submodule.1.submodule.0 (encoder block 3)
    "backbone.down2.maxpool_conv.1.double_conv.0.weight": "model.1.submodule.1.submodule.0.conv.0.conv.weight",
    "backbone.down2.maxpool_conv.1.double_conv.0.bias":   "model.1.submodule.1.submodule.0.conv.0.conv.bias",
    "backbone.down2.maxpool_conv.1.double_conv.1.weight": "model.1.submodule.1.submodule.0.conv.0.adn.N.weight",
    "backbone.down2.maxpool_conv.1.double_conv.1.bias":   "model.1.submodule.1.submodule.0.conv.0.adn.N.bias",
    "backbone.down2.maxpool_conv.1.double_conv.3.weight": "model.1.submodule.1.submodule.0.conv.1.conv.weight",
    "backbone.down2.maxpool_conv.1.double_conv.3.bias":   "model.1.submodule.1.submodule.0.conv.1.conv.bias",
    "backbone.down2.maxpool_conv.1.double_conv.4.weight": "model.1.submodule.1.submodule.0.conv.1.adn.N.weight",
    "backbone.down2.maxpool_conv.1.double_conv.4.bias":   "model.1.submodule.1.submodule.0.conv.1.adn.N.bias",

    # down3 → model.1.submodule.1.submodule.1.submodule.0 (encoder block 4)
    "backbone.down3.maxpool_conv.1.double_conv.0.weight": "model.1.submodule.1.submodule.1.submodule.0.conv.0.conv.weight",
    "backbone.down3.maxpool_conv.1.double_conv.0.bias":   "model.1.submodule.1.submodule.1.submodule.0.conv.0.conv.bias",
    "backbone.down3.maxpool_conv.1.double_conv.1.weight": "model.1.submodule.1.submodule.1.submodule.0.conv.0.adn.N.weight",
    "backbone.down3.maxpool_conv.1.double_conv.1.bias":   "model.1.submodule.1.submodule.1.submodule.0.conv.0.adn.N.bias",
    "backbone.down3.maxpool_conv.1.double_conv.3.weight": "model.1.submodule.1.submodule.1.submodule.0.conv.1.conv.weight",
    "backbone.down3.maxpool_conv.1.double_conv.3.bias":   "model.1.submodule.1.submodule.1.submodule.0.conv.1.conv.bias",
    "backbone.down3.maxpool_conv.1.double_conv.4.weight": "model.1.submodule.1.submodule.1.submodule.0.conv.1.adn.N.weight",
    "backbone.down3.maxpool_conv.1.double_conv.4.bias":   "model.1.submodule.1.submodule.1.submodule.0.conv.1.adn.N.bias",

    # down4 → model.1.submodule.1.submodule.1.submodule.1.submodule (bottleneck)
    "backbone.down4.maxpool_conv.1.double_conv.0.weight": "model.1.submodule.1.submodule.1.submodule.1.submodule.conv.0.conv.weight",
    "backbone.down4.maxpool_conv.1.double_conv.0.bias":   "model.1.submodule.1.submodule.1.submodule.1.submodule.conv.0.conv.bias",
    "backbone.down4.maxpool_conv.1.double_conv.1.weight": "model.1.submodule.1.submodule.1.submodule.1.submodule.conv.0.adn.N.weight",
    "backbone.down4.maxpool_conv.1.double_conv.1.bias":   "model.1.submodule.1.submodule.1.submodule.1.submodule.conv.0.adn.N.bias",
    "backbone.down4.maxpool_conv.1.double_conv.3.weight": "model.1.submodule.1.submodule.1.submodule.1.submodule.conv.1.conv.weight",
    "backbone.down4.maxpool_conv.1.double_conv.3.bias":   "model.1.submodule.1.submodule.1.submodule.1.submodule.conv.1.conv.bias",
    "backbone.down4.maxpool_conv.1.double_conv.4.weight": "model.1.submodule.1.submodule.1.submodule.1.submodule.conv.1.adn.N.weight",
    "backbone.down4.maxpool_conv.1.double_conv.4.bias":   "model.1.submodule.1.submodule.1.submodule.1.submodule.conv.1.adn.N.bias",
}

def load_ssl_encoder(model, ssl_ckpt_path, device):
    """
    Load COVER SSL pretrained encoder weights into AttentionUnet encoder.
    Uses an explicit layer-by-layer mapping derived from printing both
    models' named_parameters(). Maps all 5 encoder blocks (inc + down1-4).
    """
    print(f"Loading SSL encoder from {ssl_ckpt_path} ...")
    ckpt        = torch.load(ssl_ckpt_path, map_location=device)
    ssl_weights = ckpt.get("state_dict", ckpt)
    ssl_weights = {k.replace("module.", ""): v for k, v in ssl_weights.items()}

    att_state = model.state_dict()
    loaded, skipped = 0, 0

    for cover_key, att_key in COVER_TO_ATTUNET.items():
        if cover_key in ssl_weights and att_key in att_state:
            src = ssl_weights[cover_key]
            dst = att_state[att_key]
            if src.shape == dst.shape:
                att_state[att_key] = src
                loaded += 1
            else:
                print(f"  ⚠️  Shape mismatch: {cover_key} {src.shape} → {att_key} {dst.shape}")
                skipped += 1
        else:
            missing = cover_key if cover_key not in ssl_weights else att_key
            print(f"  ⚠️  Key not found: {missing}")
            skipped += 1

    model.load_state_dict(att_state)
    print(f"  Loaded : {loaded}/{len(COVER_TO_ATTUNET)} encoder layers ✅")
    if skipped > 0:
        print(f"  Skipped: {skipped} layers")
    return model

# ─── CHECKPOINT ───────────────────────────────────────────────────────────────

def save_checkpoint(state, is_best, save_dir, modelname, keep_last=3):
    os.makedirs(save_dir, exist_ok=True)
    epoch    = state["epoch"] - 1
    filename = os.path.join(save_dir,
                            f"checkpoint_{modelname}_{epoch:04d}.pth.tar")
    torch.save(state, filename)

    if is_best:
        best_path = os.path.join(save_dir, f"model_best_{modelname}.pth.tar")
        shutil.copyfile(filename, best_path)
        print(f"  ✅ New best model saved (Dice: {state['best_dice']:.4f})")

    # Remove old checkpoints
    existing = sorted(glob.glob(
        os.path.join(save_dir, f"checkpoint_{modelname}_*.pth.tar")))
    for old in existing[:-keep_last]:
        os.remove(old)

# ─── LOG WRITER ───────────────────────────────────────────────────────────────

class LogWriter:
    def __init__(self, path):
        self.path = path + ".csv"
        with open(self.path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Epoch", "Train Loss", "Val Loss",
                "Val Dice", "Val Hausdorff95"
            ])

    def write(self, row):
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow(row)

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu}"
                          if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Preprocessing ────────────────────────────────────────────────────────
    if args.npz_path and os.path.isfile(args.npz_path):
        print(f"Loading preprocessed data from {args.npz_path} ...")
    else:
        npz_path = os.path.join(args.save_dir, "seg_preprocessed.npz")
        print("Preprocessing volumes ...")
        preprocess_volumes_to_npz(
            args.image_dir, args.mask_dir, npz_path,
            img_size=args.img_size)
        args.npz_path = npz_path

    # Load shared cache
    print("Loading .npz into RAM ...")
    npz         = np.load(args.npz_path)
    all_img_keys = sorted([k.replace("img_", "") for k in npz.files
                            if k.startswith("img_")])
    shared_cache = {k: npz[k] for k in npz.files}
    npz.close()
    ram_gb = sum(v.nbytes for v in shared_cache.values()) / 1e9
    print(f"  {len(all_img_keys)} slices — {ram_gb:.2f} GB in RAM")

    # Patient-level train/val split
    patients     = sorted(set("_".join(k.split("_")[:3]) for k in all_img_keys))
    train_pats, val_pats = train_test_split(patients, test_size=0.2, random_state=42)
    train_pats   = set(train_pats)

    train_keys = [k for k in all_img_keys
                  if "_".join(k.split("_")[:3]) in train_pats]
    val_keys   = [k for k in all_img_keys
                  if "_".join(k.split("_")[:3]) not in train_pats]

    print(f"  Train: {len(train_keys)} slices ({len(train_pats)} patients)")
    print(f"  Val  : {len(val_keys)} slices ({len(val_pats)} patients)")

    # Datasets + loaders
    train_ds = SliceDataset(train_keys, shared_cache,
                            img_size=args.img_size, augment=True)
    val_ds   = SliceDataset(val_keys,   shared_cache,
                            img_size=args.img_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.workers,
                              pin_memory=True)

    # ── Model ────────────────────────────────────────────────────────────────
    model = AttentionUnet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params / 1e6:.2f} M")

    # Load SSL encoder weights if provided
    if args.ssl_checkpoint and os.path.isfile(args.ssl_checkpoint):
        model = load_ssl_encoder(model, args.ssl_checkpoint, device)

    # ── Loss + Metrics ───────────────────────────────────────────────────────
    criterion    = CombinedLoss(cldice_weight=0.3, cldice_start_epoch=20).to(device)
    dice_metric  = DiceMetric(include_background=False, reduction="mean")
    hd95_metric  = HausdorffDistanceMetric(include_background=False,
                                           percentile=95, reduction="mean")

    # ── Optimizer — two-phase ────────────────────────────────────────────────
    # Phase 1: freeze encoder, train decoder only
    # Phase 2: unfreeze all with differential LR
    def get_optimizer(phase=1):
        if phase == 1:
            # Freeze encoder (first half of model parameters)
            params = model.named_parameters()
            encoder_params = []
            decoder_params = []
            for name, p in params:
                if any(x in name for x in ['encode', 'down', 'input_block']):
                    p.requires_grad = False
                    encoder_params.append(p)
                else:
                    p.requires_grad = True
                    decoder_params.append(p)
            print(f"Phase 1 — encoder frozen, training decoder only")
            return torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=args.lr)
        else:
            # Unfreeze all with differential LR
            for p in model.parameters():
                p.requires_grad = True
            print(f"Phase 2 — full model unfrozen")
            return torch.optim.AdamW([
                {'params': [p for n, p in model.named_parameters()
                            if any(x in n for x in ['encode', 'down'])],
                 'lr': args.lr * 0.1},   # encoder — low LR
                {'params': [p for n, p in model.named_parameters()
                            if not any(x in n for x in ['encode', 'down'])],
                 'lr': args.lr},          # decoder — full LR
            ])

    optimizer = get_optimizer(phase=1 if args.ssl_checkpoint else 2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── Resume from checkpoint ───────────────────────────────────────────────
    start_epoch = 0
    best_dice   = 0.0
    logwriter   = LogWriter(os.path.join(args.save_dir, args.modelname))

    ckpt_pattern = os.path.join(args.save_dir,
                                f"checkpoint_{args.modelname}_*.pth.tar")
    existing_ckpts = sorted(glob.glob(ckpt_pattern))
    if existing_ckpts:
        ckpt_path  = existing_ckpts[-1]
        print(f"=> loading checkpoint '{ckpt_path}'")
        ckpt       = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"]
        best_dice   = ckpt.get("best_dice", 0.0)
        print(f"=> resumed from epoch {start_epoch}")
    else:
        print("=> no checkpoint found — starting from scratch")

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):

        # Switch to phase 2 after freeze_epochs
        if args.ssl_checkpoint and epoch == args.freeze_epochs:
            optimizer = get_optimizer(phase=2)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - epoch, eta_min=1e-6)

        # ── Train ────────────────────────────────────────────────────────────
        criterion.current_epoch = epoch  # update phased loss
        if epoch == 20:
            print("Loss phase B — adding clDice for topology refinement")
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{args.epochs}",
                    leave=False, mininterval=5)

        for img, msk in pbar:
            img = img.to(device, non_blocking=True)
            msk = msk.to(device, non_blocking=True)

            # GPU augmentation
            img, msk = gpu_augment(img, msk, device)

            pred = model(img)
            loss = criterion(pred, msk)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss /= len(train_loader)
        scheduler.step()

        # ── Validate ─────────────────────────────────────────────────────────
        val_loss = 0.0
        if (epoch + 1) % args.val_interval == 0:
            model.eval()
            dice_metric.reset()
            hd95_metric.reset()

            with torch.no_grad():
                for img, msk in tqdm(val_loader,
                                     desc=f"Val   Epoch {epoch+1}",
                                     leave=False, mininterval=5):
                    img  = img.to(device, non_blocking=True)
                    msk  = msk.to(device, non_blocking=True)
                    pred = model(img)

                    val_loss += criterion(pred, msk).item()

                    # Binarise predictions
                    pred_bin = (torch.sigmoid(pred) > 0.5).float()
                    dice_metric(y_pred=pred_bin, y=msk)
                    hd95_metric(y_pred=pred_bin, y=msk)

            val_loss  /= len(val_loader)
            val_dice   = dice_metric.aggregate().item()
            val_hd95   = hd95_metric.aggregate().item()
            dice_metric.reset()
            hd95_metric.reset()

            is_best  = val_dice > best_dice
            best_dice = max(val_dice, best_dice)

            print(f"\nEpoch [{epoch+1}/{args.epochs}]")
            print(f"  Train Loss : {train_loss:.4f}")
            print(f"  Val Loss   : {val_loss:.4f}")
            print(f"  Val Dice   : {val_dice:.4f}  (best: {best_dice:.4f})")
            print(f"  Val HD95   : {val_hd95:.2f} mm")

            logwriter.write([epoch+1, train_loss, val_loss,
                             val_dice, val_hd95])

            save_checkpoint(
                {
                    "epoch":      epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer":  optimizer.state_dict(),
                    "best_dice":  best_dice,
                },
                is_best=is_best,
                save_dir=args.save_dir,
                modelname=args.modelname,
            )

            # Kaggle dataset backup every N epochs
            if (KAGGLEHUB_AVAILABLE
                    and args.kaggle_dataset
                    and (epoch + 1) % args.backup_every == 0):
                try:
                    print(f"Backing up to {args.kaggle_dataset} ...")
                    kagglehub.dataset_upload(
                        args.kaggle_dataset,
                        args.save_dir,
                        version_notes=f"epoch {epoch+1} dice {val_dice:.4f}"
                    )
                    print("✅ Backup complete")
                except Exception as e:
                    print(f"\u26a0\ufe0f  Backup failed: {type(e).__name__}: {e}")
                    print(f"   Make sure '{args.kaggle_dataset}' exists on kaggle.com/datasets")
                    print(f"   Checkpoints still saved locally in {args.save_dir}")

    print(f"\nTraining complete — best Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()
