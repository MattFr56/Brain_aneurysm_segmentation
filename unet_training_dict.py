import csv
import logging
import os
import sys
from glob import glob

import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split

import monai
from monai.losses import TverskyLoss
from monai.data import CacheDataset, pad_list_data_collate, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.networks.nets import AttentionUnet
from monai.transforms import (
    Activations, AsDiscrete, Compose,
    EnsureChannelFirstd, Lambdad, LoadImaged, Orientationd,
    RandCropByPosNegLabeld, Rand3DElasticd, RandFlipd,
    RandGaussianNoised, RandRotate90d, RandScaleIntensityd,
    RandShiftIntensityd, RandZoomd, ScaleIntensityRanged, Spacingd,
    SpatialPadd, ToTensord,
)
from monai.visualize import plot_2d_or_3d_image

import functools
import builtins
builtins.print = functools.partial(builtins.print, flush=True)
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# ── Paths ──────────────────────────────────────────────────────────────────────
IMAGE_DIR  = "/kaggle/working/merged/volumes"
MASK_DIR   = "/kaggle/working/merged/labels"
CHECKPOINT = "/kaggle/working/best_model.pth"
CSV_PATH   = "/kaggle/working/training_log.csv"
CURVE_PATH = "/kaggle/working/training_curves.png"

# ── Architecture ───────────────────────────────────────────────────────────────
CHANNELS = (64, 128, 256, 512)
STRIDES  = (2, 2, 2)

# ── Hyperparameters ────────────────────────────────────────────────────────────
HU_MIN         = 100
HU_MAX         = 400
SPATIAL_SIZE   = (128, 128, 32)
NUM_EPOCHS     = 100        # ← was 500
VAL_INTERVAL   = 2
TRAIN_SAMPLES  = 4
BATCH_SIZE     = 1
SW_OVERLAP     = 0.25
SW_OVERLAP_VAL = 0.5
PATIENCE       = 40         # ← was 60
PRED_THRESHOLD = 0.4
WARMUP_EPOCHS  = 10
ACCUM_STEPS    = 4
TTA_INTERVAL   = 10         # ← new: TTA only every 10 epochs


# ── TTA ────────────────────────────────────────────────────────────────────────
TTA_FLIPS = [[], [2], [3], [4], [2,3], [2,4], [3,4], [2,3,4]]

def tta_inference(model, inputs, roi_size, overlap):
    preds = []
    for axes in TTA_FLIPS:
        x    = torch.flip(inputs, axes) if axes else inputs
        pred = sliding_window_inference(
            x, roi_size=roi_size, sw_batch_size=8,
            predictor=model, overlap=overlap,
        )
        if axes:
            pred = torch.flip(pred, axes)
        preds.append(torch.sigmoid(pred))
    return torch.mean(torch.stack(preds), dim=0)


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
    axes[0].set_title("Train Loss (Tversky)")
    axes[0].set_xlabel("Epoch"); axes[0].grid(True, alpha=0.3)

    axes[1].plot(val_epochs, dices, color="darkorange", linewidth=1.5,
                 marker="o", markersize=3)
    if dices:
        best_idx = dices.index(max(dices))
        axes[1].axvline(val_epochs[best_idx], color="red", linestyle="--",
                        alpha=0.6, label=f"Best {max(dices):.4f} @ ep {val_epochs[best_idx]}")
        axes[1].legend(fontsize=8)
    axes[1].set_title("Val Dice (TTA)"); axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1); axes[1].grid(True, alpha=0.3)

    valid_hd = [(e, h) for e, h in zip(val_epochs, hd95s) if h is not None]
    if valid_hd:
        hd_ep, hd_vals = zip(*valid_hd)
        axes[2].plot(hd_ep, hd_vals, color="seagreen", linewidth=1.5,
                     marker="o", markersize=3)
        best_hd_idx = hd_vals.index(min(hd_vals))
        axes[2].axvline(hd_ep[best_hd_idx], color="red", linestyle="--",
                        alpha=0.6, label=f"Best {min(hd_vals):.2f}mm @ ep {hd_ep[best_hd_idx]}")
        axes[2].legend(fontsize=8)
    axes[2].set_title("Val HD95 (mm) ↓"); axes[2].set_xlabel("Epoch")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  ✓ curves → {out_path}")


def main():
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # ── Data ───────────────────────────────────────────────────────────────────
    images = sorted(glob(os.path.join(IMAGE_DIR, "*.nii*")))
    segs   = sorted(glob(os.path.join(MASK_DIR,  "*.nii*")))
    assert len(images) == len(segs), f"Mismatch: {len(images)} vs {len(segs)}"
    
    all_files = []
    skipped   = []
    for i, s in zip(images, segs):
        hdr       = nib.load(i).header
        z_spacing = float(hdr.get_zooms()[2])
        if z_spacing > 2.0:
            skipped.append(os.path.basename(i))
            continue
        all_files.append({"img": i, "seg": s})
    
    print(f"✓ {len(all_files)} usable | {len(skipped)} skipped")
    for name in skipped:
        print(f"  SKIP {name}")

    train_files, val_files = train_test_split(
        all_files, test_size=0.2, random_state=42, shuffle=True
    )
    print(f"✓ Train: {len(train_files)} | Val: {len(val_files)}")

    print("=== First 3 train pairs ===")
    for d in train_files[:3]:
        print(f"  img: {os.path.basename(d['img'])} | "
              f"seg: {os.path.basename(d['seg'])}")

    # ── Transforms ─────────────────────────────────────────────────────────────
    train_transforms = Compose([
        LoadImaged(keys=["img", "seg"]),
        EnsureChannelFirstd(keys=["img", "seg"]),
        Orientationd(keys=["img", "seg"], axcodes="RAS"),
        Spacingd(keys=["img", "seg"], pixdim=(1.0, 1.0, 1.0),
                 mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["img"],
                             a_min=HU_MIN, a_max=HU_MAX,
                             b_min=0.0, b_max=1.0, clip=True),
        Lambdad(keys=["seg"], func=lambda x: (x > 0).float()),
        RandCropByPosNegLabeld(
            keys=["img", "seg"], label_key="seg",
            spatial_size=SPATIAL_SIZE,
            pos=3, neg=1,
            num_samples=TRAIN_SAMPLES,
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

    val_transforms = Compose([
        LoadImaged(keys=["img", "seg"]),
        EnsureChannelFirstd(keys=["img", "seg"]),
        Orientationd(keys=["img", "seg"], axcodes="RAS"),
        Spacingd(keys=["img", "seg"], pixdim=(1.0, 1.0, 1.0),
                 mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["img"],
                             a_min=HU_MIN, a_max=HU_MAX,
                             b_min=0.0, b_max=1.0, clip=True),
        Lambdad(keys=["seg"], func=lambda x: (x > 0).float()),
        ToTensord(keys=["img", "seg"]),
    ])

    # ── Datasets ───────────────────────────────────────────────────────────────
    print("Caching datasets...")
    train_ds = CacheDataset(data=train_files, transform=train_transforms,
                            cache_rate=1.0, num_workers=4)
    val_ds   = CacheDataset(data=val_files,   transform=val_transforms,
                            cache_rate=1.0, num_workers=2)

    # ── Sanity check ───────────────────────────────────────────────────────────
    val_check = monai.utils.misc.first(
        DataLoader(val_ds, batch_size=1, collate_fn=pad_list_data_collate)
    )
    print(f"Val img shape : {val_check['img'].shape}")
    print(f"Val seg unique: {val_check['seg'].unique()}")
    print(f"Val seg fg vox: {(val_check['seg']>0).sum().item()}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, collate_fn=pad_list_data_collate,
                              pin_memory=torch.cuda.is_available())
    val_loader   = DataLoader(val_ds, batch_size=1, num_workers=2,
                              collate_fn=pad_list_data_collate)

    # ── Model ──────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = AttentionUnet(
        spatial_dims=3, in_channels=1, out_channels=1,
        channels=CHANNELS, strides=STRIDES,
    ).to(device)

    if torch.cuda.device_count() > 1:
        print(f"  DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    # ── Loss ───────────────────────────────────────────────────────────────────
    loss_function = TverskyLoss(
        sigmoid=True,
        alpha=0.3,
        beta=0.7,
    )

    # ── Optimizer + scheduler ──────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=2e-4, weight_decay=1e-4
    )
    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.1, total_iters=WARMUP_EPOCHS
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS - WARMUP_EPOCHS, eta_min=1e-6
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[WARMUP_EPOCHS]
    )
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10
    )

    # ── Metrics ────────────────────────────────────────────────────────────────
    dice_metric = DiceMetric(include_background=False, reduction="mean",
                             get_not_nans=False)
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95,
                                          reduction="mean", get_not_nans=False)
    post_trans = Compose([
        AsDiscrete(threshold=PRED_THRESHOLD),
    ])

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
        print(f"epoch {epoch + 1}/{NUM_EPOCHS}")
        model.train()
        epoch_loss = 0
        step       = 0
        optimizer.zero_grad()

        for batch_idx, batch_data in enumerate(train_loader):
            step   += 1
            inputs  = batch_data["img"].to(device)
            labels  = batch_data["seg"].to(device)
            outputs = model(inputs)
            loss    = loss_function(outputs, labels) / ACCUM_STEPS
            loss.backward()

            if (batch_idx + 1) % ACCUM_STEPS == 0 or \
               (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * ACCUM_STEPS
            epoch_len   = max(len(train_ds) // train_loader.batch_size, 1)
            print(f"  {step}/{epoch_len}  loss: {loss.item()*ACCUM_STEPS:.4f}")
            writer.add_scalar("train_loss", loss.item() * ACCUM_STEPS,
                              epoch_len * epoch + step)

        epoch_loss /= step
        current_lr  = scheduler.get_last_lr()[0]
        scheduler.step()
        writer.add_scalar("lr", current_lr, epoch + 1)
        print(f"epoch {epoch+1} avg loss: {epoch_loss:.4f}  lr: {current_lr:.2e}")

        # ── Validation with TTA ────────────────────────────────────────────────
        # ── Validation ────────────────────────────────────────────────────────
        val_dice = val_hd95 = None
        if (epoch + 1) % VAL_INTERVAL == 0:
            model.eval()
            use_tta = ((epoch + 1) % TTA_INTERVAL == 0)
            print(f"  {'TTA ' if use_tta else 'fast'} validation...", flush=True)

            with torch.no_grad():
                for val_idx, val_data in enumerate(val_loader):
                    print(f"  val {val_idx+1}/{len(val_loader)}...", flush=True)
                    val_inputs     = val_data["img"].to(device)
                    val_labels     = val_data["seg"].to(device)

                    if use_tta:
                        mean_prob = tta_inference(
                            model, val_inputs, SPATIAL_SIZE, SW_OVERLAP_VAL
                        )
                    else:
                        out       = sliding_window_inference(
                            val_inputs, roi_size=SPATIAL_SIZE,
                            sw_batch_size=8, predictor=model,
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

            plateau_scheduler.step(val_dice)
            writer.add_scalar("val_mean_dice", val_dice, epoch + 1)
            writer.add_scalar("val_hd95",      val_hd95, epoch + 1)

            if val_dice > best_metric:
                best_metric       = val_dice
                best_metric_epoch = epoch + 1
                no_improve_count  = 0
                raw_model = model.module if hasattr(model, "module") else model
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
                print("  ✓ saved new best Dice model", flush=True)
            else:
                no_improve_count += 1
                print(f"  no improvement {no_improve_count}/{PATIENCE}", flush=True)

            if val_hd95 < best_hd95:
                best_hd95       = val_hd95
                best_hd95_epoch = epoch + 1
                raw_model = model.module if hasattr(model, "module") else model
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
                print(f"  ✓ saved new best HD95 ({best_hd95:.2f}mm)", flush=True)

            print(f"  val dice: {val_dice:.4f}  hd95: {val_hd95:.2f}mm  |  "
                  f"best dice: {best_metric:.4f} @ ep {best_metric_epoch}  |  "
                  f"best hd95: {best_hd95:.2f}mm @ ep {best_hd95_epoch}",
                  flush=True)
            plot_2d_or_3d_image(val_inputs,  epoch+1, writer, index=0, tag="image")
            plot_2d_or_3d_image(val_labels,  epoch+1, writer, index=0, tag="label")
            plot_2d_or_3d_image(
                [o.unsqueeze(0) for o in val_outputs],
                epoch+1, writer, index=0, tag="output"
            )

        append_csv(CSV_PATH, epoch+1, epoch_loss, val_dice, val_hd95, current_lr)
        plot_curves(CSV_PATH, CURVE_PATH)

        if no_improve_count >= PATIENCE:
            print(f"\nEarly stopping — no improvement for {PATIENCE} val checks")
            break

    print(f"\nTraining complete.")
    print(f"Best Dice : {best_metric:.4f} at epoch {best_metric_epoch}")
    print(f"Best HD95 : {best_hd95:.2f}mm at epoch {best_hd95_epoch}")
    writer.close()


if __name__ == "__main__":
    main()
