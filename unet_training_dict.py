import csv
import logging
import os
import re
import sys
from glob import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split

import monai
from monai.data import pad_list_data_collate
from monai.transforms import CropForegroundd
from monai.losses import DiceFocalLoss
from monai.data import CacheDataset, list_data_collate, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.networks.nets import AttentionUnet
from monai.transforms import (
    Activations, AsDiscrete, Compose, EnsureChannelFirstd, Lambdad,
    LoadImaged, Orientationd, RandCropByPosNegLabeld, Rand3DElasticd,
    RandFlipd, RandGaussianNoised, RandRotate90d, RandScaleIntensityd,
    RandShiftIntensityd, ScaleIntensityRanged, Spacingd, SpatialPadd, ToTensord,
)
from monai.visualize import plot_2d_or_3d_image

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# ── Paths ──────────────────────────────────────────────────────────────────────
IMAGE_DIR       = "/kaggle/input/datasets/mattfr56/aneursym-set-b/final_volumes"
MASK_DIR        = "/kaggle/input/datasets/mattfr56/aneursym-set-b/final_labels"
PRETRAINED_CKPT = "/kaggle/working/best_metric_model-3.pth"  # ← your .pth
CHECKPOINT      = "/kaggle/working/best_finetune_model.pth"
CSV_PATH        = "/kaggle/working/finetune_log.csv"
CURVE_PATH      = "/kaggle/working/finetune_curves.png"

# ── Architecture — must match pretrained .pth exactly ─────────────────────────
CHANNELS = (64, 128, 256, 512)
STRIDES  = (2, 2, 2)

# ── Hyperparameters ────────────────────────────────────────────────────────────
SPATIAL_SIZE   = (128, 128, 32)
TRAIN_SAMPLES  = 2
BATCH_SIZE     = 1
SW_OVERLAP     = 0.25
PRED_THRESHOLD = 0.4
VAL_INTERVAL   = 2
PATIENCE       = 50

# ── Fine-tuning phases ─────────────────────────────────────────────────────────
PHASES = [
    {"epochs": 15, "lr": 1e-3, "unfreeze": "head"},       # adapt output head
    {"epochs": 50, "lr": 1e-4, "unfreeze": "decoder"},    # decoder + attention gates
    {"epochs": 35, "lr": 1e-5, "unfreeze": "all"},        # full model, careful
]
NUM_EPOCHS = sum(p["epochs"] for p in PHASES)  # 60 total


# ── Freezing helpers ───────────────────────────────────────────────────────────
def freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False

def unfreeze_head(model):
    """model.2 — final 1x1 conv, weight + bias only."""
    freeze_all(model)
    for name, p in model.named_parameters():
        if name.startswith("model.2"):
            p.requires_grad = True
            print(f"    unfrozen: {name}")

def unfreeze_decoder(model):
    """
    Decoder = attention gates + upconv + merge blocks at each level.
    These all live in model.1.* but NOT in model.1.submodule.X.submodule.*
    i.e. the upsampling path back toward the surface.
    """
    freeze_all(model)
    decoder_patterns = [
        "model.2",                                      # head
        "model.1.attention", "model.1.upconv",          # decoder level 1
        "model.1.merge",
        "model.1.submodule.1.attention",                # decoder level 2
        "model.1.submodule.1.upconv",
        "model.1.submodule.1.merge",
        "model.1.submodule.1.submodule.1.attention",    # decoder level 3
        "model.1.submodule.1.submodule.1.upconv",
        "model.1.submodule.1.submodule.1.merge",
    ]
    unfrozen = 0
    for name, p in model.named_parameters():
        if any(name.startswith(pat) for pat in decoder_patterns):
            p.requires_grad = True
            unfrozen += 1
    print(f"    unfrozen {unfrozen} decoder params")

def unfreeze_all(model):
    unfrozen = 0
    for p in model.parameters():
        p.requires_grad = True
        unfrozen += 1
    print(f"    unfrozen all {unfrozen} params")

def apply_phase(model, optimizer, phase_cfg):
    {"head": unfreeze_head, "decoder": unfreeze_decoder,
     "all": unfreeze_all}[phase_cfg["unfreeze"]](model)

    trainable = [p for p in model.parameters() if p.requires_grad]
    if len(trainable) == 0:
        raise RuntimeError(f"Phase '{phase_cfg['unfreeze']}': 0 trainable params")

    optimizer.param_groups[0]["params"] = trainable
    optimizer.param_groups[0]["lr"]     = phase_cfg["lr"]
    n = sum(p.numel() for p in trainable)
    print(f"  unfreeze='{phase_cfg['unfreeze']}' | lr={phase_cfg['lr']} | trainable: {n:,}")
```
# ── CSV helpers ────────────────────────────────────────────────────────────────
def init_csv(path):
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "phase", "train_loss", "val_dice", "val_hd95", "lr"])

def append_csv(path, epoch, phase, train_loss, val_dice, val_hd95, lr):
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow([
            epoch, phase, f"{train_loss:.6f}",
            f"{val_dice:.6f}" if val_dice is not None else "",
            f"{val_hd95:.4f}" if val_hd95 is not None else "",
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
                hd95s.append(float(row["val_hd95"]) if row["val_hd95"] else None)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(epochs, losses, color="steelblue", linewidth=1.5)
    axes[0].set_title("Train Loss"); axes[0].set_xlabel("Epoch"); axes[0].grid(True, alpha=0.3)

    axes[1].plot(val_epochs, dices, color="darkorange", linewidth=1.5, marker="o", markersize=3)
    if dices:
        best_idx = dices.index(max(dices))
        axes[1].axvline(val_epochs[best_idx], color="red", linestyle="--", alpha=0.6,
                        label=f"Best {max(dices):.4f} @ ep {val_epochs[best_idx]}")
        axes[1].legend(fontsize=8)
    axes[1].set_title("Val Dice"); axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1); axes[1].grid(True, alpha=0.3)

    valid_hd = [(e, h) for e, h in zip(val_epochs, hd95s) if h is not None]
    if valid_hd:
        hd_ep, hd_vals = zip(*valid_hd)
        axes[2].plot(hd_ep, hd_vals, color="seagreen", linewidth=1.5, marker="o", markersize=3)
        best_hd_idx = hd_vals.index(min(hd_vals))
        axes[2].axvline(hd_ep[best_hd_idx], color="red", linestyle="--", alpha=0.6,
                        label=f"Best {min(hd_vals):.2f}mm @ ep {hd_ep[best_hd_idx]}")
        axes[2].legend(fontsize=8)
    axes[2].set_title("Val HD95 (mm) ↓"); axes[2].set_xlabel("Epoch"); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  ✓ curves → {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # ── Data ───────────────────────────────────────────────────────────────────
    images = sorted(glob(os.path.join(IMAGE_DIR, "*.nii*")))
    segs   = sorted(glob(os.path.join(MASK_DIR,  "*.nii*")))
    assert len(images) == len(segs), f"Mismatch: {len(images)} vs {len(segs)}"
    print(f"✓ {len(images)} image/mask pairs")

    # shuffle=True, no stratify needed (labels are all binary now)
    train_images, val_images, train_masks, val_masks = train_test_split(
        images, segs, test_size=0.2, random_state=42, shuffle=True
    )
    print(f"✓ Train: {val_images} | Val: {val_masks}")

    train_files = [{"img": i, "seg": s} for i, s in zip(train_images, train_masks)]
    val_files   = [{"img": i, "seg": s} for i, s in zip(val_images,   val_masks)]

    # ── Transforms (identical to your original) ────────────────────────────────
    train_transforms = Compose([
        LoadImaged(keys=["img", "seg"]),
        EnsureChannelFirstd(keys=["img", "seg"]),
        Orientationd(keys=["img", "seg"], axcodes="RAS"),
        Spacingd(keys=["img", "seg"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        CropForegroundd(keys=["img", "seg"], source_key="img"),
        ScaleIntensityRanged(keys=["img"], a_min=100, a_max=600,
                             b_min=0.0, b_max=1.0, clip=True),
        Lambdad(keys=["seg"], func=lambda x: (x > 0).float()),
        SpatialPadd(keys=["img", "seg"], spatial_size=SPATIAL_SIZE, mode="constant"),
        RandCropByPosNegLabeld(keys=["img", "seg"], label_key="seg",
                               spatial_size=SPATIAL_SIZE, pos=0.9, neg=0.1,
                               num_samples=TRAIN_SAMPLES, allow_smaller=False),
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
    val_transforms = Compose([
        LoadImaged(keys=["img", "seg"]),
        EnsureChannelFirstd(keys=["img", "seg"]),
        Orientationd(keys=["img", "seg"], axcodes="RAS"),
        Spacingd(keys=["img", "seg"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        CropForegroundd(keys=["img", "seg"], source_key="img"),
        ScaleIntensityRanged(keys=["img"], a_min=100, a_max=600,
                             b_min=0.0, b_max=1.0, clip=True),
        Lambdad(keys=["seg"], func=lambda x: (x > 0).float()),
        SpatialPadd(keys=["img", "seg"], spatial_size=SPATIAL_SIZE, mode="constant"),
        ToTensord(keys=["img", "seg"]),
    ])

    print("Caching datasets...")
    train_ds = CacheDataset(data=train_files, transform=train_transforms,
                            cache_rate=1.0, num_workers=4)
    val_ds   = CacheDataset(data=val_files,   transform=val_transforms,
                            cache_rate=1.0, num_workers=2)

    # Sanity check
    check_loader = DataLoader(train_ds, batch_size=2, num_workers=0,
                              collate_fn=pad_list_data_collate)
    check_data   = monai.utils.misc.first(check_loader)
    img, seg     = check_data["img"], check_data["seg"]
    vessel_vox   = img[seg > 0]
    print(f"Sanity — img: {img.shape} | seg unique: {seg.unique()}")
    print(f"Vessel intensity mean={vessel_vox.mean():.3f} std={vessel_vox.std():.3f}")
    print(f"Vessel voxels: {(seg>0).sum().item()}/{seg.numel()} "
          f"({100*(seg>0).float().mean().item():.2f}%)")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, collate_fn=list_data_collate,
                              pin_memory=torch.cuda.is_available())
    val_loader   = DataLoader(val_ds, batch_size=1, num_workers=2,
                              collate_fn=list_data_collate)

    # ── Model ──────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = AttentionUnet(
        spatial_dims=3, in_channels=1, out_channels=1,
        channels=CHANNELS, strides=STRIDES,
    ).to(device)

    # Load pretrained — strict=False handles any minor key mismatches
    if not os.path.exists(PRETRAINED_CKPT):
        raise FileNotFoundError(f"Checkpoint not found: {PRETRAINED_CKPT}")
    ckpt  = torch.load(PRETRAINED_CKPT, map_location=device, weights_only=False)
    #print("=== Model layer names ===")
    #for name, _ in model.named_parameters():
    #print(f"  {name}")
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"✓ Pretrained weights loaded")
    if missing:    print(f"  Missing (random-init): {missing}")
    if unexpected: print(f"  Unexpected (ignored):  {unexpected}")

    if torch.cuda.device_count() > 1:
        print(f"  DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    # ── Loss ───────────────────────────────────────────────────────────────────
    loss_function = DiceFocalLoss(
        sigmoid=True, gamma=2.0,
        lambda_dice=0.5, lambda_focal=0.5,
    )

    # ── Optimizer (params updated per phase via apply_phase) ───────────────────
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=PHASES[0]["lr"], weight_decay=1e-5
    )
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=8
    )

    # ── Metrics ────────────────────────────────────────────────────────────────
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95,
                                          reduction="mean", get_not_nans=False)
    post_trans  = Compose([
        Activations(sigmoid=True),
        AsDiscrete(threshold=PRED_THRESHOLD),
    ])

    init_csv(CSV_PATH)
    writer            = SummaryWriter()
    best_metric       = -1
    best_metric_epoch = -1
    no_improve_count  = 0
    global_epoch      = 0

    # ── Phase loop ─────────────────────────────────────────────────────────────
    for phase_idx, phase_cfg in enumerate(PHASES):
        print(f"\n{'='*50}")
        print(f"PHASE {phase_idx+1}/{len(PHASES)} — "
              f"unfreeze={phase_cfg['unfreeze']} | "
              f"epochs={phase_cfg['epochs']} | lr={phase_cfg['lr']}")
        print('='*50)

        apply_phase(model, optimizer, phase_cfg)

        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=phase_cfg["epochs"], eta_min=phase_cfg["lr"] / 100
        )

        for local_epoch in range(phase_cfg["epochs"]):
            global_epoch += 1
            print(f"\n--- epoch {global_epoch}/{NUM_EPOCHS} "
                  f"(phase {phase_idx+1}, local {local_epoch+1}/{phase_cfg['epochs']}) ---")

            # Train
            model.train()
            epoch_loss = 0
            step       = 0
            for batch_data in train_loader:
                step   += 1
                inputs  = batch_data["img"].to(device)
                labels  = batch_data["seg"].to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss    = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                epoch_len   = max(len(train_ds) // train_loader.batch_size, 1)
                print(f"  {step}/{epoch_len}  loss: {loss.item():.4f}")
                writer.add_scalar("train_loss", loss.item(), epoch_len * global_epoch + step)

            epoch_loss /= step
            current_lr  = cosine_scheduler.get_last_lr()[0]
            cosine_scheduler.step()
            writer.add_scalar("lr", current_lr, global_epoch)
            print(f"  avg loss: {epoch_loss:.4f}  lr: {current_lr:.2e}")

            # Validation
            val_dice = val_hd95 = None
            if global_epoch % VAL_INTERVAL == 0:
                model.eval()
                with torch.no_grad():
                    for val_data in val_loader:
                        val_inputs  = val_data["img"].to(device)
                        val_labels  = val_data["seg"].to(device)
                        val_outputs = sliding_window_inference(
                            val_inputs, roi_size=SPATIAL_SIZE, sw_batch_size=4,
                            predictor=model, overlap=SW_OVERLAP,
                        )
                        val_outputs    = [post_trans(i) for i in decollate_batch(val_outputs)]
                        val_labels_dec = [i for i in decollate_batch(val_labels)]
                        dice_metric(y_pred=val_outputs, y=val_labels_dec)
                        hd95_metric(y_pred=val_outputs, y=val_labels_dec)

                val_dice = dice_metric.aggregate().item()
                val_hd95 = hd95_metric.aggregate().item()
                dice_metric.reset()
                hd95_metric.reset()

                plateau_scheduler.step(val_dice)
                writer.add_scalar("val_mean_dice", val_dice, global_epoch)
                writer.add_scalar("val_hd95",      val_hd95, global_epoch)

                if val_dice > best_metric:
                    best_metric       = val_dice
                    best_metric_epoch = global_epoch
                    no_improve_count  = 0
                    # Unwrap DataParallel before saving
                    raw_model = model.module if hasattr(model, "module") else model
                    torch.save({
                        "state_dict":   raw_model.state_dict(),
                        "channels":     CHANNELS,
                        "strides":      STRIDES,
                        "spatial_size": SPATIAL_SIZE,
                        "epoch":        global_epoch,
                        "best_dice":    best_metric,
                        "best_hd95":    val_hd95,
                        "threshold":    PRED_THRESHOLD,
                    }, CHECKPOINT)
                    print("  ✓ saved new best model")
                else:
                    no_improve_count += 1
                    print(f"  no improvement {no_improve_count}/{PATIENCE}")

                print(f"  val dice: {val_dice:.4f}  hd95: {val_hd95:.2f}mm  |  "
                      f"best: {best_metric:.4f} @ epoch {best_metric_epoch}")

            append_csv(CSV_PATH, global_epoch, phase_idx + 1,
                       epoch_loss, val_dice, val_hd95, current_lr)
            plot_curves(CSV_PATH, CURVE_PATH)

            if no_improve_count >= PATIENCE:
                print(f"\nEarly stopping triggered after {PATIENCE} val checks")
                break

        if no_improve_count >= PATIENCE:
            break

    print(f"\nFine-tuning complete.")
    print(f"Best Dice : {best_metric:.4f} at epoch {best_metric_epoch}")
    print(f"CSV       → {CSV_PATH}")
    print(f"Plot      → {CURVE_PATH}")
    writer.close()


if __name__ == "__main__":
    main()
