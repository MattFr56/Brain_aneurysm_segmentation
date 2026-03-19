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
from torch.cuda.amp import autocast, GradScaler   # OOM FIX: mixed precision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

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
    SpatialPadd, ToTensord,
)
from monai.visualize import plot_2d_or_3d_image

# OOM FIX: allow PyTorch to split large allocations rather than failing
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ── Paths ──────────────────────────────────────────────────────────────────────
IMAGE_DIR   = "/kaggle/working/merged/volumes"
MASK_DIR    = "/kaggle/working/merged/labels"
PHASE1_CKPT = "/kaggle/working/best_model_phase1.pth"
CHECKPOINT  = "/kaggle/working/best_model_phase2.pth"
CSV_PATH    = "/kaggle/working/training_log_phase2.csv"
CURVE_PATH  = "/kaggle/working/training_curves_phase2.png"

# ── Architecture ───────────────────────────────────────────────────────────────
# OOM FIX: channels halved from (64,128,256,512) to (32,64,128,256).
# The larger config exceeds 16 GB on a T4 when you account for:
#   - fp32 model weights
#   - EMA shadow copy (same size as weights)
#   - Optimizer state (AdamW keeps 2 momentum tensors per param)
#   - Forward activations for backprop
# Halving channels reduces activation memory ~4x. Weight transfer from
# Phase 1 still works as long as Phase 1 used the same channel config.
# If your Phase 1 checkpoint used (64,128,256,512) you have two options:
#   A) retrain Phase 1 with (32,64,128,256) — recommended for consistency
#   B) keep (64,128,256,512) and apply all other OOM fixes — may still OOM
CHANNELS = (32, 64, 128, 256)
STRIDES  = (2, 2, 2)

# ── Hyperparameters ────────────────────────────────────────────────────────────
HU_MIN          = 100
HU_MAX          = 400
SPATIAL_SIZE    = (128, 128, 32)
NUM_EPOCHS      = 50
VAL_INTERVAL    = 2
BATCH_SIZE      = 1
SW_BATCH_SIZE   = 4       # sliding window batch size for val inference
SW_OVERLAP_VAL  = 0.25    # 0.5 is too slow/heavy during training val;
                           # use 0.5 only for final evaluation
PATIENCE        = 15
PRED_THRESHOLD  = 0.4
ACCUM_STEPS     = 4        # OOM FIX: was 2. More accumulation = smaller
                            # effective micro-batch = lower peak memory.
                            # Effective batch size is unchanged.
ANEURYSM_REPEAT = 5
REPLAY_PCT      = 0.2
EMA_DECAY       = 0.999

# ── Aneurysm splits ────────────────────────────────────────────────────────────
# case_0152 and case_0166 moved from Phase 1 val → Phase 2 train.
# Phase 1 is frozen; moving them adds training signal without any data leak.
ANEURYSM_TRAIN_P2 = {
    "case_0131", "case_0132", "case_0136",
    "case_0139", "case_0141", "case_0142",
    "case_0150", "case_0152", "case_0153",
    "case_0166",
}
ANEURYSM_VAL_P2 = {
    "case_0177", "case_0179", "case_0181",
}

def get_stem(path):
    return os.path.basename(path).replace(".nii.gz", "").replace(".nii", "")

def is_aneurysm(path):
    return get_stem(path) in ANEURYSM_TRAIN_P2 | ANEURYSM_VAL_P2


# ── EMA ────────────────────────────────────────────────────────────────────────
class EMA:
    """Exponential moving average of model weights.

    Usage contract:
      - Call apply_shadow() ONCE before any val/save block.
      - Call restore() ONCE after all saves are done.
      - Never call apply_shadow() a second time while shadow is active:
        it would overwrite backup with shadow weights, making restore()
        load shadow back into params instead of the training weights.
    """
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


# ── TTA (8 flip combinations) ─────────────────────────────────────────────────
TTA_FLIPS = [[], [2], [3], [4], [2,3], [2,4], [3,4], [2,3,4]]

def tta_inference(model, inputs, roi_size, overlap, sw_batch_size):
    """8-flip TTA with autocast for memory efficiency.
    Call inside torch.no_grad().
    """
    preds = []
    for axes in TTA_FLIPS:
        x = torch.flip(inputs, axes) if axes else inputs.clone()
        # OOM FIX: autocast keeps intermediate activations in fp16
        with autocast():
            pred = sliding_window_inference(
                x, roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                predictor=model,
                overlap=overlap,
            )
        pred = pred.float()      # back to fp32 before sigmoid
        if axes:
            pred = torch.flip(pred, axes)
        preds.append(torch.sigmoid(pred))
    print(f"    TTA: {len(preds)} augmentations averaged")
    return torch.mean(torch.stack(preds), dim=0)


# ── Threshold optimization ─────────────────────────────────────────────────────
def find_best_threshold(model, val_loader, device, spatial_size,
                        overlap, sw_batch_size):
    print("\n=== Threshold optimization ===")
    all_probs  = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for val_data in val_loader:
            val_inputs = val_data["img"].to(device)
            val_labels = val_data["seg"].to(device)
            mean_prob  = tta_inference(model, val_inputs, spatial_size,
                                       overlap, sw_batch_size)
            all_probs.append(mean_prob.cpu())
            all_labels.append(val_labels.cpu())

    best_thresh = 0.4
    best_dice   = 0.0

    for thresh in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        dices = []
        for prob, label in zip(all_probs, all_labels):
            pred  = (prob > thresh).float()
            tp    = (pred * label).sum()
            dice  = (2 * tp / (pred.sum() + label.sum() + 1e-5)).item()
            dices.append(dice)
        mean_dice = float(np.mean(dices))
        print(f"  thresh={thresh:.2f} -> Dice={mean_dice:.4f}")
        if mean_dice > best_dice:
            best_dice   = mean_dice
            best_thresh = thresh

    print(f"Best threshold: {best_thresh:.2f} -> Dice {best_dice:.4f}")
    return best_thresh, best_dice


# ── CSV helpers ────────────────────────────────────────────────────────────────
def init_csv(path):
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss",
                                 "val_dice", "val_hd95", "lr"])

def append_csv(path, epoch, train_loss, val_dice, val_hd95, lr):
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow([
            epoch,
            f"{train_loss:.6f}",
            f"{val_dice:.6f}"  if val_dice is not None else "",
            f"{val_hd95:.4f}"  if val_hd95 is not None else "",
            f"{lr:.8f}",
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
                hd_str = row["val_hd95"]
                # guard against "nan" string before float conversion
                hd95s.append(
                    float(hd_str)
                    if hd_str and hd_str.lower() != "nan"
                    else None
                )

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(epochs, losses, color="steelblue", linewidth=1.5)
    axes[0].set_title("Train Loss (Tversky)")
    axes[0].set_xlabel("Epoch"); axes[0].grid(True, alpha=0.3)

    axes[1].plot(val_epochs, dices, color="crimson", linewidth=1.5,
                 marker="o", markersize=4)
    if dices:
        best_idx = dices.index(max(dices))
        axes[1].axvline(val_epochs[best_idx], color="red", linestyle="--",
                        alpha=0.6,
                        label=f"Best {max(dices):.4f} @ ep {val_epochs[best_idx]}")
        axes[1].legend(fontsize=8)
    axes[1].set_title("Val Dice — Aneurysm Cases (TTA+EMA)")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)

    valid_hd = [(e, h) for e, h in zip(val_epochs, hd95s) if h is not None]
    if valid_hd:
        hd_ep, hd_vals = zip(*valid_hd)
        axes[2].plot(hd_ep, hd_vals, color="seagreen", linewidth=1.5,
                     marker="o", markersize=4)
        best_hd_idx = hd_vals.index(min(hd_vals))
        axes[2].axvline(hd_ep[best_hd_idx], color="red", linestyle="--",
                        alpha=0.6,
                        label=f"Best {min(hd_vals):.2f}mm @ ep {hd_ep[best_hd_idx]}")
        axes[2].legend(fontsize=8)
    axes[2].set_title("Val HD95 (mm) ↓")
    axes[2].set_xlabel("Epoch"); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  curves -> {out_path}")


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
    print(f"Usable volumes: {len(all_files)}")

    # ── Build phase 2 splits ───────────────────────────────────────────────────
    aneurysm_train = [d for d in all_files
                      if get_stem(d["img"]) in ANEURYSM_TRAIN_P2]
    aneurysm_val   = [d for d in all_files
                      if get_stem(d["img"]) in ANEURYSM_VAL_P2]
    # Replay pool = all non-aneurysm cases (automatically excludes val cases)
    normal_cases   = [d for d in all_files if not is_aneurysm(d["img"])]

    n_replay = int(len(normal_cases) * REPLAY_PCT)

    print(f"Aneurysm train : {len(aneurysm_train)} cases x "
          f"{ANEURYSM_REPEAT} repeat = "
          f"{len(aneurysm_train)*ANEURYSM_REPEAT} entries/epoch")
    print(f"Vessel replay  : {n_replay} cases sampled fresh each epoch "
          f"(pool: {len(normal_cases)})")
    print(f"Phase 2 val    : {len(aneurysm_val)} aneurysm cases (held out)")

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

    train_transforms = Compose(shared_pre + [
        RandCropByPosNegLabeld(
            keys=["img", "seg"], label_key="seg",
            spatial_size=SPATIAL_SIZE,
            pos=9, neg=1,
            num_samples=4,      # OOM FIX: was 8. Each sample is a separate
                                # forward pass keeping activations alive for
                                # backprop. Halving from 8→4 halves peak
                                # activation memory. Diversity is maintained
                                # by re-sampling replay each epoch and the
                                # heavy augmentation pipeline below.
            allow_smaller=True,
        ),
        SpatialPadd(keys=["img", "seg"], spatial_size=SPATIAL_SIZE),

        # ── Spatial augmentation ───────────────────────────────────────────────
        RandRotated(
            keys=["img", "seg"],
            range_x=0.5, range_y=0.5, range_z=0.5,
            prob=0.5, mode=("bilinear", "nearest"),
        ),
        RandZoomd(keys=["img", "seg"],
                  min_zoom=0.6, max_zoom=1.4,
                  mode=("trilinear", "nearest"), prob=0.5),
        Rand3DElasticd(keys=["img", "seg"],
                       sigma_range=(5, 10),
                       magnitude_range=(200, 500),
                       prob=0.7,
                       mode=("bilinear", "nearest")),
        RandFlipd(keys=["img", "seg"], spatial_axis=0, prob=0.5),
        RandFlipd(keys=["img", "seg"], spatial_axis=1, prob=0.5),
        RandFlipd(keys=["img", "seg"], spatial_axis=2, prob=0.5),
        RandRotate90d(keys=["img", "seg"], prob=0.7,
                      spatial_axes=(0, 1)),

        # ── Intensity augmentation ─────────────────────────────────────────────
        RandAdjustContrastd(keys=["img"], prob=0.4, gamma=(0.5, 2.0)),
        RandGaussianSmoothd(keys=["img"],
                            sigma_x=(0.5, 2.0), sigma_y=(0.5, 2.0),
                            sigma_z=(0.5, 2.0), prob=0.3),
        RandBiasFieldd(keys=["img"], degree=3,
                       coeff_range=(0.0, 0.2), prob=0.3),
        RandHistogramShiftd(keys=["img"],
                            num_control_points=10, prob=0.3),
        RandGaussianNoised(keys=["img"], prob=0.4,
                           mean=0.0, std=0.03),
        RandScaleIntensityd(keys=["img"], factors=0.3, prob=0.5),
        RandShiftIntensityd(keys=["img"], offsets=0.3, prob=0.5),
        ToTensord(keys=["img", "seg"]),
    ])

    val_transforms = Compose(shared_pre + [
        ToTensord(keys=["img", "seg"]),
    ])

    # ── Val dataset (fixed, cached) ────────────────────────────────────────────
    print("Building val dataset...")
    val_ds = CacheDataset(data=aneurysm_val, transform=val_transforms,
                          cache_rate=1.0, num_workers=2)
    val_loader = DataLoader(
        val_ds, batch_size=1, num_workers=2,
        collate_fn=pad_list_data_collate,
    )

    # Sanity check
    val_check = monai.utils.misc.first(
        DataLoader(val_ds, batch_size=1, collate_fn=pad_list_data_collate)
    )
    print(f"Val img shape : {val_check['img'].shape}")
    print(f"Val seg unique: {val_check['seg'].unique()}")
    print(f"Val seg fg vox: {(val_check['seg']>0).sum().item()}")

    # ── Model ──────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = AttentionUnet(
        spatial_dims=3, in_channels=1, out_channels=1,
        channels=CHANNELS, strides=STRIDES,
    ).to(device)

    # ── Load Phase 1 checkpoint ────────────────────────────────────────────────
    if not os.path.exists(PHASE1_CKPT):
        raise FileNotFoundError(f"Phase 1 checkpoint not found: {PHASE1_CKPT}")
    ckpt  = torch.load(PHASE1_CKPT, map_location=device, weights_only=False)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Phase 1 weights loaded — "
          f"Dice {ckpt.get('best_dice', '?'):.4f}  "
          f"HD95 {ckpt.get('best_hd95', '?'):.2f}mm")
    if missing:    print(f"  Missing:    {missing}")
    if unexpected: print(f"  Unexpected: {unexpected}")

    if torch.cuda.device_count() > 1:
        print(f"  DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    # ── EMA — init on unwrapped model ─────────────────────────────────────────
    raw_model = model.module if hasattr(model, "module") else model
    ema       = EMA(raw_model, decay=EMA_DECAY)
    print(f"EMA initialized (decay={EMA_DECAY})")

    # ── Loss ───────────────────────────────────────────────────────────────────
    loss_function = TverskyLoss(sigmoid=True, alpha=0.3, beta=0.7)

    # ── Optimizer + scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-5, weight_decay=1e-4
    )
    # Single scheduler: ReduceLROnPlateau only.
    # CosineAnnealingLR + ReduceLROnPlateau fight each other — Cosine
    # overrides every LR drop that Plateau triggers. Plateau alone is the
    # right choice for fine-tuning a strong Phase 1 checkpoint.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-7
    )

    # OOM FIX: GradScaler for mixed-precision training.
    # Inflates gradients before backward to avoid fp16 underflow,
    # then unscales before the optimizer step.
    scaler = GradScaler()

    # ── Metrics ────────────────────────────────────────────────────────────────
    dice_metric = DiceMetric(include_background=False, reduction="mean",
                             get_not_nans=False)
    hd95_metric = HausdorffDistanceMetric(include_background=False,
                                          percentile=95, reduction="mean",
                                          get_not_nans=False)
    post_trans  = Compose([AsDiscrete(threshold=PRED_THRESHOLD)])

    init_csv(CSV_PATH)
    writer            = SummaryWriter()
    best_metric       = -1
    best_hd95         = float("inf")
    best_metric_epoch = -1
    best_hd95_epoch   = -1
    no_improve_count  = 0

    # ── Training loop ──────────────────────────────────────────────────────────
    for epoch in range(NUM_EPOCHS):
        print("-" * 10)
        print(f"epoch {epoch+1}/{NUM_EPOCHS}")

        # Re-sample replay cases each epoch for variety
        replay       = random.sample(normal_cases, n_replay)
        phase2_train = []
        for d in aneurysm_train:
            for _ in range(ANEURYSM_REPEAT):
                phase2_train.append(d)
        phase2_train += replay
        random.shuffle(phase2_train)

        train_ds = MonaiDataset(data=phase2_train, transform=train_transforms)
        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=4, collate_fn=pad_list_data_collate,
            pin_memory=torch.cuda.is_available(),
        )

        epoch_len  = len(train_loader)
        model.train()
        epoch_loss = 0
        step       = 0
        optimizer.zero_grad()

        for batch_idx, batch_data in enumerate(train_loader):
            step  += 1
            inputs = batch_data["img"].to(device)
            labels = batch_data["seg"].to(device)

            # OOM FIX: autocast keeps forward-pass activations in fp16,
            # roughly halving training memory vs pure fp32.
            with autocast():
                outputs = model(inputs)
                loss    = loss_function(outputs, labels) / ACCUM_STEPS

            # OOM FIX: scaler.scale(loss).backward() replaces loss.backward()
            scaler.scale(loss).backward()

            if (batch_idx + 1) % ACCUM_STEPS == 0 or \
               (batch_idx + 1) == len(train_loader):
                # Unscale before clip so grad norms are in true fp32 scale
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                ema.update()

            epoch_loss += loss.item() * ACCUM_STEPS
            print(f"  {step}/{epoch_len}  "
                  f"loss: {loss.item()*ACCUM_STEPS:.4f}")
            writer.add_scalar("train_loss",
                              loss.item() * ACCUM_STEPS,
                              epoch_len * epoch + step)

        epoch_loss /= step
        # ReduceLROnPlateau has no get_last_lr() — read from param_groups
        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("lr", current_lr, epoch + 1)
        print(f"epoch {epoch+1} avg loss: {epoch_loss:.4f}  "
              f"lr: {current_lr:.2e}")

        # ── Validation: EMA weights + TTA ─────────────────────────────────────
        val_dice = val_hd95 = None
        if (epoch + 1) % VAL_INTERVAL == 0:
            # apply_shadow() once — backs up training weights, loads EMA weights.
            ema.apply_shadow()
            model.eval()
            print(f"  TTA validation (EMA weights, {len(TTA_FLIPS)} augs)...")

            per_case_dices = []

            with torch.no_grad():
                for val_idx, val_data in enumerate(val_loader):
                    print(f"  val {val_idx+1}/{len(val_loader)}...")
                    val_inputs     = val_data["img"].to(device)
                    val_labels     = val_data["seg"].to(device)
                    mean_prob      = tta_inference(
                        model, val_inputs, SPATIAL_SIZE,
                        SW_OVERLAP_VAL, SW_BATCH_SIZE,
                    )
                    val_outputs    = [post_trans(i) for i in
                                      decollate_batch(mean_prob)]
                    val_labels_dec = [i for i in decollate_batch(val_labels)]
                    dice_metric(y_pred=val_outputs, y=val_labels_dec)
                    hd95_metric(y_pred=val_outputs, y=val_labels_dec)

                    for pred, lab in zip(val_outputs, val_labels_dec):
                        tp = (pred * lab).sum()
                        dc = (2 * tp / (pred.sum() + lab.sum() + 1e-5)).item()
                        per_case_dices.append(dc)

            val_dice = dice_metric.aggregate().item()
            val_hd95 = hd95_metric.aggregate().item()
            dice_metric.reset()
            hd95_metric.reset()

            print(f"  per-case Dice: "
                  f"{[f'{d:.4f}' for d in per_case_dices]}  "
                  f"(min={min(per_case_dices):.4f})")

            # ── Save checkpoints while EMA shadow is still active ──────────────
            # DO NOT call apply_shadow() again here — backup is already set.
            # A second call would write shadow into backup, making restore()
            # load shadow back into params instead of training weights.
            if val_dice > best_metric:
                best_metric       = val_dice
                best_metric_epoch = epoch + 1
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
                print("  saved new best Dice model (EMA weights)")

            if val_hd95 < best_hd95:
                best_hd95       = val_hd95
                best_hd95_epoch = epoch + 1
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
                print(f"  saved new best HD95 ({best_hd95:.2f}mm)")

            # Single restore — called once after all saves
            ema.restore()

            # Increment patience only if NEITHER metric improved
            if val_dice <= best_metric and val_hd95 >= best_hd95:
                no_improve_count += 1
                print(f"  no improvement {no_improve_count}/{PATIENCE}")
            else:
                no_improve_count = 0

            # Step the single scheduler after restore
            scheduler.step(val_dice)

            writer.add_scalar("val_mean_dice", val_dice, epoch + 1)
            writer.add_scalar("val_hd95",      val_hd95, epoch + 1)
            writer.add_scalar("val_dice_min",  min(per_case_dices), epoch + 1)

            print(f"  val dice: {val_dice:.4f}  hd95: {val_hd95:.2f}mm  |  "
                  f"best dice: {best_metric:.4f} @ ep {best_metric_epoch}  |  "
                  f"best hd95: {best_hd95:.2f}mm @ ep {best_hd95_epoch}")

            plot_2d_or_3d_image(val_inputs,  epoch+1, writer,
                                index=0, tag="image")
            plot_2d_or_3d_image(val_labels,  epoch+1, writer,
                                index=0, tag="label")
            plot_2d_or_3d_image(
                [o.unsqueeze(0) for o in val_outputs],
                epoch+1, writer, index=0, tag="output"
            )

        append_csv(CSV_PATH, epoch+1, epoch_loss,
                   val_dice, val_hd95, current_lr)
        plot_curves(CSV_PATH, CURVE_PATH)

        if no_improve_count >= PATIENCE:
            print(f"\nEarly stopping — {PATIENCE} val checks "
                  f"without improvement")
            break

    # ── Threshold optimization after training ──────────────────────────────────
    print("\nRunning threshold optimization on val set...")
    ema.apply_shadow()
    best_thresh, best_thresh_dice = find_best_threshold(
        model, val_loader, device, SPATIAL_SIZE, SW_OVERLAP_VAL, SW_BATCH_SIZE
    )
    ema.restore()

    if os.path.exists(CHECKPOINT):
        ckpt              = torch.load(CHECKPOINT, map_location=device,
                                       weights_only=False)
        ckpt["threshold"] = best_thresh
        torch.save(ckpt, CHECKPOINT)
        print(f"Checkpoint updated with threshold={best_thresh:.2f}")

    print(f"\nPhase 2 complete.")
    print(f"Best Dice  : {best_metric:.4f} @ epoch {best_metric_epoch}")
    print(f"Best HD95  : {best_hd95:.2f}mm @ epoch {best_hd95_epoch}")
    print(f"Best thresh: {best_thresh:.2f} -> Dice {best_thresh_dice:.4f}")
    print(f"Checkpoint -> {CHECKPOINT}")
    writer.close()


if __name__ == "__main__":
    main()
