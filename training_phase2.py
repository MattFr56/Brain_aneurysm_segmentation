import csv
import logging
import os
import sys
from glob import glob
import random

import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import functools
import builtins
builtins.print = functools.partial(builtins.print, flush=True)

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split

import monai
from monai.losses import TverskyLoss
from monai.data import CacheDataset, pad_list_data_collate, decollate_batch
from monai.data import Dataset as MonaiDataset
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.networks.nets import AttentionUnet
from monai.transforms import (
    Activations, AsDiscrete, Compose,
    EnsureChannelFirstd, Lambdad, LoadImaged, Orientationd,
    RandAdjustContrastd, RandBiasFieldd, RandCropByPosNegLabeld,
    Rand3DElasticd, RandFlipd, RandGaussianNoised, RandGaussianSmoothd,
    RandHistogramShiftd, RandRotated, RandRotate90d, RandScaleIntensityd,
    RandShiftIntensityd, RandZoomd, ScaleIntensityRanged, Spacingd,
    SpatialPadd, ToTensord, MapTransform,
)
from monai.visualize import plot_2d_or_3d_image

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# ── Paths ──────────────────────────────────────────────────────────────────────
IMAGE_DIR       = "/kaggle/working/merged/volumes"
MASK_DIR        = "/kaggle/working/merged/labels"
PRETRAINED_CKPT = "/kaggle/working/best_model_phase1b.pth"
CHECKPOINT      = "/kaggle/working/best_model_phase1b2.pth"
SWA_CHECKPOINT  = "/kaggle/working/best_model_phase1b2_swa.pth"
CSV_PATH        = "/kaggle/working/training_log_phase1b2.csv"
CURVE_PATH      = "/kaggle/working/training_curves_phase1b2.png"

# ── Architecture ───────────────────────────────────────────────────────────────
CHANNELS = (64, 128, 256, 512)
STRIDES  = (2, 2, 2)

# ── Hyperparameters ────────────────────────────────────────────────────────────
HU_MIN         = 100
HU_MAX         = 400
SPATIAL_SIZE   = (128, 128, 32)
NUM_EPOCHS     = 150
VAL_INTERVAL   = 2
BATCH_SIZE     = 1
SW_OVERLAP_VAL = 0.5
PATIENCE       = 30
PRED_THRESHOLD = 0.4
ACCUM_STEPS    = 2
EMA_DECAY      = 0.999
SWA_START      = 120
RESUME_EPOCH   = 50

TRAIN_SAMPLES_STANDARD = 4
TRAIN_SAMPLES_ANEURYSM = 6
ANEURYSM_REPEAT        = 5
TTA_INTERVAL           = 10

# ── Curriculum epochs ──────────────────────────────────────────────────────────
CURRICULUM_TOPCOW_ONLY = 20
CURRICULUM_ADD_RSNA    = 50

# ── Aneurysm case splits ───────────────────────────────────────────────────────
ANEURYSM_TRAIN = {
    "case_0131", "case_0132", "case_0136",
    "case_0139", "case_0141", "case_0142",
    "case_0150", "case_0153",
}
ANEURYSM_VAL = {
    "case_0152", "case_0166",
    "case_0177", "case_0179", "case_0181",
}

# ── Aneurysm z-ranges (original volume coordinates + 10 slice margin) ─────────
ANEURYSM_Z_RANGES = {
    "case_0131": (38,  61),   # volume_04182  — 3 slices + margin
    "case_0132": (19,  42),   # volume_04305  — 3 slices + margin
    "case_0136": (38,  65),   # volume_11381
    "case_0139": (54,  86),   # volume_14733
    "case_0141": (100, 130),  # volume_15982
    "case_0142": (62,  98),   # volume_17368
    "case_0150": (125, 218),  # volume_37400  — large aneurysm
    "case_0153": (104, 143),  # volume_39466
}


def get_stem(path):
    return os.path.basename(path).replace(".nii.gz", "").replace(".nii", "")

def is_aneurysm_train(path): return get_stem(path) in ANEURYSM_TRAIN
def is_aneurysm_val(path):   return get_stem(path) in ANEURYSM_VAL
def is_aneurysm(path):       return get_stem(path) in ANEURYSM_TRAIN | ANEURYSM_VAL
def is_topcow(path):         return int(get_stem(path).split("_")[1]) <= 125
def is_rsna_normal(path):    return not is_topcow(path) and not is_aneurysm(path)


# ── Targeted aneurysm sampling mask transform ──────────────────────────────────
class CreateAneurysmSamplingMaskd(MapTransform):
    """
    Creates a binary sampling mask concentrated on aneurysm z-range.
    Forces RandCropByPosNegLabeld to center patches on aneurysm slices.
    After resampling to 1mm isotropic, z-indices are scaled proportionally.
    """
    def __init__(self, keys, z_ranges, spacing_key="seg"):
        super().__init__(keys)
        self.z_ranges    = z_ranges
        self.spacing_key = spacing_key

    def __call__(self, data):
        d    = dict(data)
        stem = get_stem(str(d.get("img", d.get("img_meta_dict", {})
                                  .get("filename_or_obj", ""))))

        seg = d["seg"]  # (1, H, W, D) after resampling

        if stem in self.z_ranges:
            z_min_orig, z_max_orig = self.z_ranges[stem]
            d_size = seg.shape[-1]  # current z size after resampling

            # Scale z-range proportionally to resampled volume
            # We use the meta information if available, otherwise
            # use the range directly (works when orig spacing ~ 1mm)
            orig_z = d.get("seg_meta_dict", {}).get(
                "spatial_shape", [None, None, d_size]
            )[-1]

            if orig_z and orig_z != d_size:
                scale   = d_size / orig_z
                z_min   = max(0, int(z_min_orig * scale))
                z_max   = min(d_size, int(z_max_orig * scale))
            else:
                z_min   = max(0, min(z_min_orig, d_size - 1))
                z_max   = min(d_size, z_max_orig)

            # Create sampling mask — 1 in aneurysm z-range
            aneu_mask = torch.zeros_like(seg)
            aneu_mask[..., z_min:z_max] = 1.0

            # Intersect with foreground to stay on vessels
            aneu_mask = aneu_mask * (seg > 0).float()

            # Fallback: if intersection is empty use full z-range
            if aneu_mask.sum() == 0:
                print(f"  WARNING: empty aneurysm mask for {stem} "
                      f"z=[{z_min},{z_max}] — using full seg")
                aneu_mask = (seg > 0).float()

            d["aneurysm_mask"] = aneu_mask
            print(f"  Aneurysm mask: {stem} z=[{z_min},{z_max}] "
                  f"fg_vox={int(aneu_mask.sum())}")
        else:
            # No z-range — use full seg as fallback
            d["aneurysm_mask"] = (seg > 0).float()

        return d


# ── Curriculum data builder ────────────────────────────────────────────────────
def get_curriculum_files(epoch, all_files_phase1):
    topcow      = [d for d in all_files_phase1 if is_topcow(d["img"])]
    rsna_normal = [d for d in all_files_phase1 if is_rsna_normal(d["img"])]
    rsna_aneu   = [d for d in all_files_phase1
                   if is_aneurysm_train(d["img"])]

    if epoch < CURRICULUM_TOPCOW_ONLY:
        files = topcow
        label = "TopCoW only"
    elif epoch < CURRICULUM_ADD_RSNA:
        files = topcow + rsna_normal
        label = "TopCoW + RSNA normal"
    else:
        aneu_repeated = rsna_aneu * ANEURYSM_REPEAT
        files         = topcow + rsna_normal + aneu_repeated
        label         = f"TopCoW + RSNA + aneurysm x{ANEURYSM_REPEAT}"

    return files, label


# ── EMA ────────────────────────────────────────────────────────────────────────
class EMA:
    def __init__(self, model, decay=0.999):
        self.model  = model
        self.decay  = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name]
                    + (1 - self.decay) * param.data
                )

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data        = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]


# ── TTA ────────────────────────────────────────────────────────────────────────
TTA_FLIPS     = [[], [2], [3], [4], [2,3], [2,4], [3,4], [2,3,4]]
TTA_ROTATIONS = [0, 2]

def tta_inference(model, inputs, roi_size, overlap):
    preds = []
    for axes in TTA_FLIPS:
        for k in TTA_ROTATIONS:
            x = torch.flip(inputs, axes) if axes else inputs.clone()
            if k > 0:
                x = torch.rot90(x, k, dims=[2, 3])
            pred = sliding_window_inference(
                x, roi_size=roi_size, sw_batch_size=2,
                predictor=model, overlap=overlap,
            )
            if k > 0:
                pred = torch.rot90(pred, -k, dims=[2, 3])
            if axes:
                pred = torch.flip(pred, axes)
            preds.append(torch.sigmoid(pred))
    return torch.mean(torch.stack(preds), dim=0)


# ── CSV helpers ────────────────────────────────────────────────────────────────
def init_csv(path):
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss",
                                 "val_dice", "val_hd95",
                                 "lr", "curriculum"])

def append_csv(path, epoch, train_loss, val_dice,
               val_hd95, lr, curriculum):
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow([
            epoch,
            f"{train_loss:.6f}",
            f"{val_dice:.6f}" if val_dice is not None else "",
            f"{val_hd95:.4f}" if val_hd95 is not None else "",
            f"{lr:.8f}",
            curriculum,
        ])

def plot_curves(csv_path, out_path):
    epochs, losses = [], []
    val_epochs, dices, hd95s = [], [], []

    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            losses.append(float(row["train_loss"]))
            if row["val_dice"]:
                val_epochs.append(int(row["epoch"]))
                dices.append(float(row["val_dice"]))
                hd95s.append(
                    float(row["val_hd95"]) if row["val_hd95"] else None
                )

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(epochs, losses, color="steelblue", linewidth=1.5)
    axes[0].axvline(CURRICULUM_TOPCOW_ONLY, color="orange",
                    linestyle="--", alpha=0.5, label="Add RSNA")
    axes[0].axvline(CURRICULUM_ADD_RSNA, color="red",
                    linestyle="--", alpha=0.5, label="Add aneurysm")
    axes[0].legend(fontsize=7)
    axes[0].set_title("Train Loss"); axes[0].set_xlabel("Epoch")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(val_epochs, dices, color="darkorange", linewidth=1.5,
                 marker="o", markersize=3)
    if dices:
        best_idx = dices.index(max(dices))
        axes[1].axvline(val_epochs[best_idx], color="red", linestyle="--",
                        alpha=0.6,
                        label=f"Best {max(dices):.4f} @ ep {val_epochs[best_idx]}")
        axes[1].legend(fontsize=8)
    axes[1].set_title("Val Dice (TTA+EMA)")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)

    valid_hd = [(e, h) for e, h in zip(val_epochs, hd95s) if h is not None]
    if valid_hd:
        hd_ep, hd_vals = zip(*valid_hd)
        axes[2].plot(hd_ep, hd_vals, color="seagreen", linewidth=1.5,
                     marker="o", markersize=3)
        best_hd_idx = hd_vals.index(min(hd_vals))
        axes[2].axvline(hd_ep[best_hd_idx], color="red", linestyle="--",
                        alpha=0.6,
                        label=f"Best {min(hd_vals):.2f}mm @ ep {hd_ep[best_hd_idx]}")
        axes[2].legend(fontsize=8)
    axes[2].set_title("Val HD95 (mm) down")
    axes[2].set_xlabel("Epoch"); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  curves -> {out_path}")


# ── Aneurysm-aware dataset ─────────────────────────────────────────────────────
class AwareDataset(MonaiDataset):
    def __init__(self, data, transform_standard, transform_aneurysm):
        super().__init__(data=data, transform=None)
        self.transform_standard = transform_standard
        self.transform_aneurysm = transform_aneurysm

    def __getitem__(self, index):
        item = self.data[index]
        if is_aneurysm_train(item["img"]):
            return self.transform_aneurysm(item)
        return self.transform_standard(item)


def main():
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # ── Data ───────────────────────────────────────────────────────────────────
    images = sorted(glob(os.path.join(IMAGE_DIR, "*.nii*")))
    segs   = sorted(glob(os.path.join(MASK_DIR,  "*.nii*")))
    assert len(images) == len(segs), f"Mismatch: {len(images)} vs {len(segs)}"

    all_files = []
    for i, s in zip(images, segs):
        hdr = nib.load(i).header
        if float(hdr.get_zooms()[2]) > 2.0:
            continue
        all_files.append({"img": i, "seg": s})
    print(f"Usable: {len(all_files)} volumes")

    # Hold out aneurysm val cases
    all_files_phase1 = [d for d in all_files
                        if not is_aneurysm_val(d["img"])]
    print(f"Phase 1b pool: {len(all_files_phase1)} "
          f"({len(all_files)-len(all_files_phase1)} aneurysm val held out)")

    # Fixed val split
    _, val_files = train_test_split(
        all_files_phase1, test_size=0.2, random_state=42, shuffle=True
    )
    seen = set()
    val_files_dedup = []
    for d in val_files:
        if d["img"] not in seen:
            seen.add(d["img"])
            val_files_dedup.append(d)
    val_files = val_files_dedup
    print(f"Val: {len(val_files)} cases (fixed)")

    # ── Transforms ─────────────────────────────────────────────────────────────
    shared_pre = [
        LoadImaged(keys=["img", "seg"]),
        EnsureChannelFirstd(keys=["img", "seg"]),
        Orientationd(keys=["img", "seg"], axcodes="RAS"),
        Spacingd(keys=["img", "seg"], pixdim=(1.0, 1.0, 1.0),
                 mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["img"],
                             a_min=HU_MIN, a_max=HU_MAX,
                             b_min=0.0, b_max=1.0, clip=True),
        Lambdad(keys=["seg"], func=lambda x: (x > 0).float()),
    ]

    train_transforms_standard = Compose(shared_pre + [
        RandCropByPosNegLabeld(
            keys=["img", "seg"], label_key="seg",
            spatial_size=SPATIAL_SIZE,
            pos=9, neg=1,
            num_samples=TRAIN_SAMPLES_STANDARD,
            allow_smaller=True,
        ),
        SpatialPadd(keys=["img", "seg"], spatial_size=SPATIAL_SIZE),
        RandZoomd(keys=["img", "seg"],
                  min_zoom=0.85, max_zoom=1.15,
                  mode=("trilinear", "nearest"), prob=0.3),
        Rand3DElasticd(keys=["img", "seg"], sigma_range=(3, 5),
                       magnitude_range=(50, 150), prob=0.3,
                       mode=("bilinear", "nearest")),
        RandFlipd(keys=["img", "seg"], spatial_axis=0, prob=0.5),
        RandFlipd(keys=["img", "seg"], spatial_axis=1, prob=0.5),
        RandFlipd(keys=["img", "seg"], spatial_axis=2, prob=0.5),
        RandRotate90d(keys=["img", "seg"], prob=0.5, spatial_axes=(0, 1)),
        RandGaussianNoised(keys=["img"], prob=0.15, mean=0.0, std=0.01),
        RandScaleIntensityd(keys=["img"], factors=0.1, prob=0.5),
        RandShiftIntensityd(keys=["img"], offsets=0.1, prob=0.5),
        ToTensord(keys=["img", "seg"]),
    ])

    train_transforms_aneurysm = Compose(shared_pre + [
        # Create targeted sampling mask from z-range
        CreateAneurysmSamplingMaskd(
            keys=["seg"],
            z_ranges=ANEURYSM_Z_RANGES,
        ),
        # Sample centered on aneurysm z-range
        RandCropByPosNegLabeld(
            keys=["img", "seg", "aneurysm_mask"],
            label_key="aneurysm_mask",  # ← targeted sampling
            spatial_size=SPATIAL_SIZE,
            pos=1, neg=0,              # always center on aneurysm
            num_samples=TRAIN_SAMPLES_ANEURYSM,
            allow_smaller=True,
        ),
        SpatialPadd(keys=["img", "seg", "aneurysm_mask"],
                    spatial_size=SPATIAL_SIZE),

        # Stronger spatial augmentation
        RandRotated(
            keys=["img", "seg"],
            range_x=0.5, range_y=0.5, range_z=0.5,
            prob=0.5, mode=("bilinear", "nearest"),
        ),
        RandZoomd(keys=["img", "seg"],
                  min_zoom=0.7, max_zoom=1.3,
                  mode=("trilinear", "nearest"), prob=0.5),
        Rand3DElasticd(keys=["img", "seg"],
                       sigma_range=(5, 10),
                       magnitude_range=(200, 500),
                       prob=0.7,
                       mode=("bilinear", "nearest")),
        RandFlipd(keys=["img", "seg"], spatial_axis=0, prob=0.5),
        RandFlipd(keys=["img", "seg"], spatial_axis=1, prob=0.5),
        RandFlipd(keys=["img", "seg"], spatial_axis=2, prob=0.5),
        RandRotate90d(keys=["img", "seg"], prob=0.7, spatial_axes=(0, 1)),

        # Stronger intensity augmentation
        RandAdjustContrastd(keys=["img"], prob=0.4, gamma=(0.5, 2.0)),
        RandGaussianSmoothd(keys=["img"],
                            sigma_x=(0.5, 2.0), sigma_y=(0.5, 2.0),
                            sigma_z=(0.5, 2.0), prob=0.3),
        RandBiasFieldd(keys=["img"], degree=3,
                       coeff_range=(0.0, 0.2), prob=0.3),
        RandHistogramShiftd(keys=["img"],
                            num_control_points=10, prob=0.3),
        RandGaussianNoised(keys=["img"], prob=0.4, mean=0.0, std=0.03),
        RandScaleIntensityd(keys=["img"], factors=0.3, prob=0.5),
        RandShiftIntensityd(keys=["img"], offsets=0.3, prob=0.5),
        ToTensord(keys=["img", "seg"]),
    ])

    val_transforms = Compose(shared_pre + [
        ToTensord(keys=["img", "seg"]),
    ])

    # ── Val dataset ────────────────────────────────────────────────────────────
    print("Caching val dataset...")
    val_ds = CacheDataset(data=val_files, transform=val_transforms,
                          cache_rate=1.0, num_workers=2)
    val_check = monai.utils.misc.first(
        DataLoader(val_ds, batch_size=1, collate_fn=pad_list_data_collate)
    )
    print(f"Val img shape : {val_check['img'].shape}")
    print(f"Val seg unique: {val_check['seg'].unique()}")
    print(f"Val seg fg vox: {(val_check['seg']>0).sum().item()}")

    val_loader = DataLoader(val_ds, batch_size=1, num_workers=2,
                            collate_fn=pad_list_data_collate)

    # ── Model ──────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = AttentionUnet(
        spatial_dims=3, in_channels=1, out_channels=1,
        channels=CHANNELS, strides=STRIDES,
    ).to(device)

    if os.path.exists(PRETRAINED_CKPT):
        ckpt  = torch.load(PRETRAINED_CKPT, map_location=device,
                           weights_only=False)
        state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        best_metric = ckpt.get("best_dice", -1)
        best_hd95   = ckpt.get("best_hd95", float("inf"))
        print(f"Resumed from Dice {best_metric:.4f} "
              f"HD95 {best_hd95:.2f}mm")
        if missing:    print(f"  Missing:    {missing}")
        if unexpected: print(f"  Unexpected: {unexpected}")
    else:
        best_metric = -1
        best_hd95   = float("inf")
        print("No checkpoint — training from scratch")

    if torch.cuda.device_count() > 1:
        print(f"  DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    # ── EMA + SWA ──────────────────────────────────────────────────────────────
    raw_model = model.module if hasattr(model, "module") else model
    ema       = EMA(raw_model, decay=EMA_DECAY)
    swa_model = AveragedModel(model)
    print(f"EMA (decay={EMA_DECAY}) + SWA (start ep {SWA_START}) ready")

    # ── Loss + mixed precision ─────────────────────────────────────────────────
    loss_function = TverskyLoss(sigmoid=True, alpha=0.3, beta=0.7)
    scaler        = GradScaler()

    # ── Optimizer + schedulers ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=2e-5, weight_decay=1e-4
    )
    cosine_scheduler = CosineAnnealingWarmRestarts(
    optimizer, T_0=30, T_mult=2, eta_min=1e-6,
    last_epoch=RESUME_EPOCH - 1  # ← tells scheduler where we are
    )
    swa_scheduler = SWALR(optimizer, swa_lr=1e-5)
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=8
    )

    # ── Metrics ────────────────────────────────────────────────────────────────
    dice_metric = DiceMetric(include_background=False, reduction="mean",
                             get_not_nans=False)
    hd95_metric = HausdorffDistanceMetric(include_background=False,
                                          percentile=95, reduction="mean",
                                          get_not_nans=False)
    post_trans  = Compose([AsDiscrete(threshold=PRED_THRESHOLD)])

    init_csv(CSV_PATH)
    writer            = SummaryWriter()
    best_metric_epoch = -1
    best_hd95_epoch   = -1
    no_improve_count  = 0
    swa_active        = False

    # ── Training loop ──────────────────────────────────────────────────────────
    for epoch in range(RESUME_EPOCH, NUM_EPOCHS):
        print("-" * 10)
        print(f"epoch {epoch+1}/{NUM_EPOCHS}")

        curr_files, curr_label = get_curriculum_files(epoch, all_files_phase1)

        # Clear cache when aneurysm data first appears
        if epoch == CURRICULUM_ADD_RSNA:
            torch.cuda.empty_cache()
            print("Cache cleared for aneurysm curriculum")
            
        train_pool = [d for d in curr_files
                      if d["img"] not in {v["img"] for v in val_files}]
        random.shuffle(train_pool)

        train_ds = AwareDataset(
            data=train_pool,
            transform_standard=train_transforms_standard,
            transform_aneurysm=train_transforms_aneurysm,
        )
        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=4, collate_fn=pad_list_data_collate,
            pin_memory=torch.cuda.is_available(),
        )
        print(f"  Curriculum: {curr_label} ({len(train_pool)} cases)")

        # SWA activation
        if epoch == SWA_START and not swa_active:
            swa_active = True
            print(f"  SWA activated at epoch {epoch+1}")

        # Train
        model.train()
        epoch_loss = 0
        step       = 0
        optimizer.zero_grad()

        for batch_idx, batch_data in enumerate(train_loader):
            step   += 1
            inputs  = batch_data["img"].to(device)
            labels  = batch_data["seg"].to(device)

            with autocast():
                outputs = model(inputs)
                loss    = loss_function(outputs, labels) / ACCUM_STEPS

            scaler.scale(loss).backward()

            if (batch_idx + 1) % ACCUM_STEPS == 0 or \
               (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                ema.update()
                if swa_active:
                    swa_model.update_parameters(model)

            epoch_loss += loss.item() * ACCUM_STEPS
            epoch_len   = max(len(train_ds) // train_loader.batch_size, 1)
            print(f"  {step}/{epoch_len}  "
                  f"loss: {loss.item()*ACCUM_STEPS:.4f}")
            writer.add_scalar("train_loss",
                              loss.item() * ACCUM_STEPS,
                              epoch_len * epoch + step)

        epoch_loss /= step

        # LR step
        if swa_active:
            swa_scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
        else:
            cosine_scheduler.step()
            current_lr = cosine_scheduler.get_last_lr()[0]

        writer.add_scalar("lr", current_lr, epoch + 1)
        print(f"epoch {epoch+1} avg loss: {epoch_loss:.4f} "
              f"lr: {current_lr:.2e} "
              f"{'[SWA]' if swa_active else ''}")

        # ── Validation ─────────────────────────────────────────────────────────
        val_dice = val_hd95 = None
        if (epoch + 1) % VAL_INTERVAL == 0:
            use_tta = ((epoch + 1) % TTA_INTERVAL == 0)
            ema.apply_shadow()
            model.eval()
            print(f"  {'TTA' if use_tta else 'fast'} val (EMA)...")

            with torch.no_grad():
                for val_idx, val_data in enumerate(val_loader):
                    print(f"  val {val_idx+1}/{len(val_loader)}...")
                    val_inputs  = val_data["img"].to(device)
                    val_labels  = val_data["seg"].to(device)

                    if use_tta:
                        mean_prob = tta_inference(
                            model, val_inputs,
                            SPATIAL_SIZE, SW_OVERLAP_VAL
                        )
                    else:
                        with autocast():
                            out = sliding_window_inference(
                                val_inputs, roi_size=SPATIAL_SIZE,
                                sw_batch_size=4, predictor=model,
                                overlap=SW_OVERLAP_VAL,
                            )
                        mean_prob = torch.sigmoid(out)

                    val_outputs    = [post_trans(i) for i in
                                      decollate_batch(mean_prob)]
                    val_labels_dec = [i for i in decollate_batch(val_labels)]
                    dice_metric(y_pred=val_outputs, y=val_labels_dec)
                    hd95_metric(y_pred=val_outputs, y=val_labels_dec)

            val_dice = dice_metric.aggregate().item()
            val_hd95 = hd95_metric.aggregate().item()
            dice_metric.reset()
            hd95_metric.reset()
            ema.restore()

            plateau_scheduler.step(val_dice)
            writer.add_scalar("val_mean_dice", val_dice, epoch + 1)
            writer.add_scalar("val_hd95",      val_hd95, epoch + 1)

            if val_dice > best_metric:
                best_metric       = val_dice
                best_metric_epoch = epoch + 1
                no_improve_count  = 0
                ema.apply_shadow()
                raw_model = model.module if hasattr(model, "module") \
                            else model
                torch.save({
                    "state_dict":   raw_model.state_dict(),
                    "channels":     CHANNELS,
                    "strides":      STRIDES,
                    "spatial_size": SPATIAL_SIZE,
                    "epoch":        epoch + 1,
                    "best_dice":    best_metric,
                    "best_hd95":    val_hd95,
                    "threshold":    PRED_THRESHOLD,
                    "hu_min":       HU_MIN,
                    "hu_max":       HU_MAX,
                }, CHECKPOINT)
                ema.restore()
                print("  saved new best Dice model (EMA)")
            else:
                if use_tta:
                    no_improve_count += 1
                    print(f"  no improvement "
                          f"{no_improve_count}/{PATIENCE}")
                else:
                    print("  fast val — not counting toward patience")

            if val_hd95 < best_hd95:
                best_hd95       = val_hd95
                best_hd95_epoch = epoch + 1
                ema.apply_shadow()
                raw_model = model.module if hasattr(model, "module") \
                            else model
                torch.save({
                    "state_dict":   raw_model.state_dict(),
                    "channels":     CHANNELS,
                    "strides":      STRIDES,
                    "spatial_size": SPATIAL_SIZE,
                    "epoch":        epoch + 1,
                    "best_dice":    val_dice,
                    "best_hd95":    best_hd95,
                    "threshold":    PRED_THRESHOLD,
                    "hu_min":       HU_MIN,
                    "hu_max":       HU_MAX,
                }, CHECKPOINT.replace(".pth", "_hd95.pth"))
                ema.restore()
                print(f"  saved new best HD95 ({best_hd95:.2f}mm)")

            print(f"  val dice: {val_dice:.4f}  hd95: {val_hd95:.2f}mm  |"
                  f"  best dice: {best_metric:.4f} @ ep {best_metric_epoch}"
                  f"  |  best hd95: {best_hd95:.2f}mm @ ep {best_hd95_epoch}")
            plot_2d_or_3d_image(val_inputs,  epoch+1, writer,
                                index=0, tag="image")
            plot_2d_or_3d_image(val_labels,  epoch+1, writer,
                                index=0, tag="label")
            plot_2d_or_3d_image(
                [o.unsqueeze(0) for o in val_outputs],
                epoch+1, writer, index=0, tag="output"
            )

        append_csv(CSV_PATH, epoch+1, epoch_loss,
                   val_dice, val_hd95, current_lr, curr_label)
        plot_curves(CSV_PATH, CURVE_PATH)

        if no_improve_count >= PATIENCE:
            print(f"\nEarly stopping — {PATIENCE} TTA val checks "
                  f"without improvement")
            break

    # ── SWA finalization ───────────────────────────────────────────────────────
    if swa_active:
        print("\nFinalizing SWA — updating BatchNorm stats...")
        update_bn(train_loader, swa_model, device=device)
        raw_swa = swa_model.module if hasattr(swa_model, "module") \
                  else swa_model
        torch.save({
            "state_dict":   raw_swa.state_dict(),
            "channels":     CHANNELS,
            "strides":      STRIDES,
            "spatial_size": SPATIAL_SIZE,
            "epoch":        NUM_EPOCHS,
            "threshold":    PRED_THRESHOLD,
            "hu_min":       HU_MIN,
            "hu_max":       HU_MAX,
        }, SWA_CHECKPOINT)
        print(f"SWA checkpoint -> {SWA_CHECKPOINT}")

    print(f"\nPhase 1b complete.")
    print(f"Best Dice  : {best_metric:.4f} @ epoch {best_metric_epoch}")
    print(f"Best HD95  : {best_hd95:.2f}mm @ epoch {best_hd95_epoch}")
    print(f"Checkpoint -> {CHECKPOINT}")
    if swa_active:
        print(f"SWA ckpt   -> {SWA_CHECKPOINT}")
    writer.close()


if __name__ == "__main__":
    main()
