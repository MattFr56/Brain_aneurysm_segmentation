import csv
import logging
import os
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
from monai.losses import DiceFocalLoss
from monai.data import CacheDataset, list_data_collate, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.networks.nets import AttentionUnet
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    EnsureChannelFirstd,
    Lambdad,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    Rand3DElasticd,
    RandFlipd,
    RandGaussianNoised,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    ToTensord,
)
from monai.visualize import plot_2d_or_3d_image

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# ── Paths ──────────────────────────────────────────────────────────────────────
IMAGE_DIR  = "/kaggle/input/datasets/mattfr56/aneursym-set-b/final_volumes"
MASK_DIR   = "/kaggle/input/datasets/mattfr56/aneursym-set-b/final_labels"
CHECKPOINT = "/kaggle/working/best_metric_model_4.pth"
CSV_PATH   = "/kaggle/working/training_log.csv"
CURVE_PATH = "/kaggle/working/training_curves.png"

# ── Architecture ───────────────────────────────────────────────────────────────
# MODE = "resume"  → resume 0.81 checkpoint, channels=(32,64,128,256)
# MODE = "wider"   → train from scratch, wider channels=(48,96,192,384)
MODE = "wider"

if MODE == "resume":
    CHANNELS = (32, 64, 128, 256)
    STRIDES  = (2, 2, 2)
else:
    CHANNELS = (64, 128, 256, 512)
    STRIDES  = (2, 2, 2)

# ── Hyperparameters ────────────────────────────────────────────────────────────
SPATIAL_SIZE    = (128, 128, 32)
NUM_EPOCHS      = 500
VAL_INTERVAL    = 2
TRAIN_SAMPLES   = 2
BATCH_SIZE      = 1
SW_OVERLAP      = 0.25
PATIENCE        = 60
PRED_THRESHOLD  = 0.4    # lower than 0.5 — better recall on thin vessels


# ── CSV helpers ────────────────────────────────────────────────────────────────
def init_csv(path):
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_dice", "val_hd95", "lr"])

def append_csv(path, epoch, train_loss, val_dice, val_hd95, lr):
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow([
            epoch,
            f"{train_loss:.6f}",
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
    axes[0].set_title("Train Loss (DiceFocal)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(val_epochs, dices, color="darkorange", linewidth=1.5, marker="o", markersize=3)
    if dices:
        best_idx = dices.index(max(dices))
        axes[1].axvline(val_epochs[best_idx], color="red", linestyle="--", alpha=0.6,
                        label=f"Best {max(dices):.4f} @ ep {val_epochs[best_idx]}")
        axes[1].legend(fontsize=8)
    axes[1].set_title("Val Dice (include_background=False)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)

    valid_hd = [(e, h) for e, h in zip(val_epochs, hd95s) if h is not None]
    if valid_hd:
        hd_ep, hd_vals = zip(*valid_hd)
        axes[2].plot(hd_ep, hd_vals, color="seagreen", linewidth=1.5, marker="o", markersize=3)
        best_hd_idx = hd_vals.index(min(hd_vals))
        axes[2].axvline(hd_ep[best_hd_idx], color="red", linestyle="--", alpha=0.6,
                        label=f"Best {min(hd_vals):.2f}mm @ ep {hd_ep[best_hd_idx]}")
        axes[2].legend(fontsize=8)
    axes[2].set_title("Val HD95 (mm) ↓ lower is better")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("HD95 (mm)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  ✓ curves saved → {out_path}")


def main():
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    print(f"MODE: {MODE} | channels: {CHANNELS} | spatial_size: {SPATIAL_SIZE}")

    # ── Data ───────────────────────────────────────────────────────────────────
    images = sorted(glob(os.path.join(IMAGE_DIR, "*.nii*")))
    segs   = sorted(glob(os.path.join(MASK_DIR,  "*.nii*")))
    assert len(images) == len(segs), f"Mismatch: {len(images)} vs {len(segs)}"
    print(f"✓ {len(images)} image/mask pairs — paired by sorted order")

    train_images, val_images, train_masks, val_masks = train_test_split(
        images, segs, test_size=0.2, random_state=42
    )
    print(f"✓ {val_images}")
    train_files = [{"img": img, "seg": seg} for img, seg in zip(train_images, train_masks)]
    val_files   = [{"img": img, "seg": seg} for img, seg in zip(val_images,   val_masks)]

    # ── Transforms ─────────────────────────────────────────────────────────────
    train_transforms = Compose([
        LoadImaged(keys=["img", "seg"]),
        EnsureChannelFirstd(keys=["img", "seg"]),
        Orientationd(keys=["img", "seg"], axcodes="RAS"),
        Spacingd(keys=["img", "seg"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(
            keys=["img"],
            a_min=100, a_max=600,
            b_min=0.0, b_max=1.0,
            clip=True,
        ),
        Lambdad(keys=["seg"], func=lambda x: (x > 0).float()),
        SpatialPadd(keys=["img", "seg"], spatial_size=SPATIAL_SIZE),
        RandCropByPosNegLabeld(
            keys=["img", "seg"],
            label_key="seg",
            spatial_size=SPATIAL_SIZE,
            pos=0.9,
            neg=0.1,
            num_samples=TRAIN_SAMPLES,
            allow_smaller=False,
        ),
        Rand3DElasticd(
            keys=["img", "seg"],
            sigma_range=(3, 5),
            magnitude_range=(50, 150),
            prob=0.3,
            mode=("bilinear", "nearest"),
        ),
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
        ScaleIntensityRanged(
            keys=["img"],
            a_min=100, a_max=600,
            b_min=0.0, b_max=1.0,
            clip=True,
        ),
        Lambdad(keys=["seg"], func=lambda x: (x > 0).float()),
        ToTensord(keys=["img", "seg"]),
    ])

    # ── CacheDataset — faster epochs ───────────────────────────────────────────
    print("Caching datasets...")
    train_ds = CacheDataset(
        data=train_files,
        transform=train_transforms,
        cache_rate=1.0,
        num_workers=4,
    )
    val_ds = CacheDataset(
        data=val_files,
        transform=val_transforms,
        cache_rate=1.0,
        num_workers=2,
    )

    # Sanity check on first cached batch
    check_loader = DataLoader(train_ds, batch_size=2, num_workers=0, collate_fn=list_data_collate)
    check_data   = monai.utils.misc.first(check_loader)
    img, seg     = check_data["img"], check_data["seg"]
    vessel_vox   = img[seg > 0]
    print(f"Sanity — img: {img.shape} | seg unique: {seg.unique()}")
    print(f"Vessel intensity mean={vessel_vox.mean():.3f} std={vessel_vox.std():.3f}")
    print(f"Vessel voxels: {(seg>0).sum().item()}/{seg.numel()} "
          f"({100*(seg>0).float().mean().item():.2f}%)")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=2, collate_fn=list_data_collate)

    # ── Model ──────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = AttentionUnet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=CHANNELS,
        strides=STRIDES,
    ).to(device)

    # ── Resume from checkpoint ─────────────────────────────────────────────────
    best_metric = -1
    if MODE == "resume" and os.path.exists(CHECKPOINT):
        ckpt  = torch.load(CHECKPOINT, map_location=device, weights_only=False)
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        model.load_state_dict(state)
        best_metric = (
            ckpt.get("best_dice", ckpt.get("best_metric", -1))
            if isinstance(ckpt, dict) else -1
        )
        print(f"✓ Resumed checkpoint — best metric restored to {best_metric:.4f}")
    elif MODE == "wider":
        print("✓ Wider model — training from scratch")
    else:
        print("No checkpoint found — training from scratch")

    # ── Loss ───────────────────────────────────────────────────────────────────
    loss_function = DiceFocalLoss(
        sigmoid=True,
        gamma=2.0,
        lambda_dice=0.5,
        lambda_focal=0.5,
    )

    # ── Optimiser + schedulers ─────────────────────────────────────────────────
    lr = 1e-4 if MODE == "resume" else 2e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
    )
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10
    )

    # ── Metrics ────────────────────────────────────────────────────────────────
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    hd95_metric = HausdorffDistanceMetric(
        include_background=False, percentile=95, reduction="mean", get_not_nans=False
    )
    post_trans = Compose([
        Activations(sigmoid=True),
        AsDiscrete(threshold=PRED_THRESHOLD),   # 0.4 — better recall on thin vessels
    ])

    # ── CSV init ───────────────────────────────────────────────────────────────
    init_csv(CSV_PATH)

    # ── Training loop ──────────────────────────────────────────────────────────
    best_metric_epoch = -1
    no_improve_count  = 0
    writer            = SummaryWriter()

    for epoch in range(NUM_EPOCHS):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{NUM_EPOCHS}")
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
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)

        epoch_loss /= step
        current_lr  = cosine_scheduler.get_last_lr()[0]
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}  lr: {current_lr:.2e}")
        cosine_scheduler.step()
        writer.add_scalar("lr", current_lr, epoch + 1)

        # ── Validation ────────────────────────────────────────────────────────
        val_dice = None
        val_hd95 = None

        if (epoch + 1) % VAL_INTERVAL == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs = val_data["img"].to(device)
                    val_labels = val_data["seg"].to(device)

                    val_outputs = sliding_window_inference(
                        val_inputs,
                        roi_size=SPATIAL_SIZE,
                        sw_batch_size=4,
                        predictor=model,
                        overlap=SW_OVERLAP,
                    )
                    val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                    val_labels  = [i for i in decollate_batch(val_labels)]

                    dice_metric(y_pred=val_outputs, y=val_labels)
                    hd95_metric(y_pred=val_outputs, y=val_labels)

                val_dice = dice_metric.aggregate().item()
                val_hd95 = hd95_metric.aggregate().item()
                dice_metric.reset()
                hd95_metric.reset()

                # ReduceLROnPlateau step on val Dice
                plateau_scheduler.step(val_dice)

                if val_dice > best_metric:
                    best_metric       = val_dice
                    best_metric_epoch = epoch + 1
                    no_improve_count  = 0
                    torch.save({
                        "state_dict":   model.state_dict(),
                        "channels":     CHANNELS,
                        "strides":      STRIDES,
                        "spatial_size": SPATIAL_SIZE,
                        "epoch":        epoch + 1,
                        "best_dice":    best_metric,
                        "best_hd95":    val_hd95,
                        "threshold":    PRED_THRESHOLD,
                    }, CHECKPOINT)
                    print("  ✓ saved new best model")
                else:
                    no_improve_count += 1
                    print(f"  no improvement {no_improve_count}/{PATIENCE}")

                print(
                    f"  val dice: {val_dice:.4f}  hd95: {val_hd95:.2f}mm  |  "
                    f"best dice: {best_metric:.4f} @ epoch {best_metric_epoch}"
                )
                writer.add_scalar("val_mean_dice", val_dice, epoch + 1)
                writer.add_scalar("val_hd95",      val_hd95, epoch + 1)
                plot_2d_or_3d_image(val_inputs,  epoch + 1, writer, index=0, tag="image")
                plot_2d_or_3d_image(val_labels,  epoch + 1, writer, index=0, tag="label")
                plot_2d_or_3d_image(val_outputs, epoch + 1, writer, index=0, tag="output")

        # ── CSV + curves ───────────────────────────────────────────────────────
        append_csv(CSV_PATH, epoch + 1, epoch_loss, val_dice, val_hd95, current_lr)
        plot_curves(CSV_PATH, CURVE_PATH)

        # ── Early stopping ─────────────────────────────────────────────────────
        if no_improve_count >= PATIENCE:
            print(f"\nEarly stopping — no improvement for {PATIENCE} val checks "
                  f"({PATIENCE * VAL_INTERVAL} epochs)")
            break

    print(f"\nTraining complete.")
    print(f"Best Dice : {best_metric:.4f} at epoch {best_metric_epoch}")
    print(f"CSV       → {CSV_PATH}")
    print(f"Plot      → {CURVE_PATH}")
    writer.close()


if __name__ == "__main__":
    main()
