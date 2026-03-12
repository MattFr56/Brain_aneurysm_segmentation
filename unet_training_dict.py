import logging
import os
import sys
from glob import glob

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split

import monai
from monai.losses import DiceFocalLoss
from monai.data import list_data_collate, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
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
    RandFlipd,
    RandGaussianNoised,
    RandRotate90d,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    ToTensord,
)
from monai.visualize import plot_2d_or_3d_image


# ── Paths ──────────────────────────────────────────────────────────────────────
IMAGE_DIR   = "/kaggle/input/datasets/mattfr56/cow-data/imagesTr"
MASK_DIR    = "/kaggle/input/datasets/mattfr56/cow-data/labelsTr"
CHECKPOINT  = "/kaggle/working/best_metric_model.pth"

# ── Hyperparameters ────────────────────────────────────────────────────────────
SPATIAL_SIZE   = (128, 128, 32)   # must be divisible by 2^4=16 in every dim
NUM_EPOCHS     = 200
VAL_INTERVAL   = 2
TRAIN_SAMPLES  = 16               # RandCrop samples per volume
VAL_SAMPLES    = 4
BATCH_SIZE     = 2                # x TRAIN_SAMPLES patches per step
LR             = 2e-4
SW_OVERLAP     = 0.25


def main():
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # ── Data ───────────────────────────────────────────────────────────────────
    images = sorted(glob(os.path.join(IMAGE_DIR, "*_0000.nii*")))
    segs   = sorted(glob(os.path.join(MASK_DIR,  "*.nii*")))
    assert len(images) == len(segs), f"Image/mask count mismatch: {len(images)} vs {len(segs)}"
    print(f"Found {len(images)} volumes")

    train_images, val_images, train_masks, val_masks = train_test_split(
        images, segs, test_size=0.2, random_state=42
    )
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
            a_min=-1000, a_max=500,
            b_min=0.0,   b_max=1.0,
            clip=True,
        ),
        # Cast AFTER resampling so nearest-neighbour stays on float, then cast
        Lambdad(keys=["seg"], func=lambda x: (x > 0).float()),   # float for DiceFocalLoss
        SpatialPadd(keys=["img", "seg"], spatial_size=SPATIAL_SIZE),
        RandCropByPosNegLabeld(
            keys=["img", "seg"],
            label_key="seg",
            spatial_size=SPATIAL_SIZE,
            pos=0.7,
            neg=0.3,
            num_samples=TRAIN_SAMPLES,
            allow_smaller=False,
        ),
        RandFlipd(keys=["img", "seg"], spatial_axis=0, prob=0.5),
        RandFlipd(keys=["img", "seg"], spatial_axis=1, prob=0.5),
        RandFlipd(keys=["img", "seg"], spatial_axis=2, prob=0.5),
        RandRotate90d(keys=["img", "seg"], prob=0.5, spatial_axes=(0, 1)),
        RandGaussianNoised(keys=["img"], prob=0.15, mean=0.0, std=0.01),
        ToTensord(keys=["img", "seg"]),
    ])

    val_transforms = Compose([
        LoadImaged(keys=["img", "seg"]),
        EnsureChannelFirstd(keys=["img", "seg"]),
        Orientationd(keys=["img", "seg"], axcodes="RAS"),
        Spacingd(keys=["img", "seg"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(
            keys=["img"],
            a_min=-1000, a_max=500,
            b_min=0.0,   b_max=1.0,
            clip=True,
        ),
        Lambdad(keys=["seg"], func=lambda x: (x > 0).float()),
        ToTensord(keys=["img", "seg"]),
        # No cropping for val — sliding_window_inference handles full volumes
    ])

    # ── Sanity check ───────────────────────────────────────────────────────────
    check_ds     = monai.data.Dataset(data=train_files[:2], transform=train_transforms)
    check_loader = DataLoader(check_ds, batch_size=2, num_workers=2, collate_fn=list_data_collate)
    check_data   = monai.utils.misc.first(check_loader)
    print("Sanity check — img:", check_data["img"].shape,
          "| seg:", check_data["seg"].shape,
          "| seg unique:", check_data["seg"].unique())

    # ── DataLoaders ────────────────────────────────────────────────────────────
    train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )

    val_ds     = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=2, collate_fn=list_data_collate)

    # ── Model ──────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionUnet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
    ).to(device)

    # ── Loss, optimiser, scheduler ─────────────────────────────────────────────
    loss_function = DiceFocalLoss(
        sigmoid=True,          # model outputs raw logits
        gamma=2.0,
        lambda_dice=0.5,
        lambda_focal=0.5,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
    )

    # ── Metric + post-processing ───────────────────────────────────────────────
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    post_trans  = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    # ── Training loop ──────────────────────────────────────────────────────────
    best_metric       = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values     = []
    writer            = SummaryWriter()

    for epoch in range(NUM_EPOCHS):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{NUM_EPOCHS}")
        model.train()
        epoch_loss = 0
        step = 0

        for batch_data in train_loader:
            step += 1
            inputs = batch_data["img"].to(device)
            labels = batch_data["seg"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss    = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_len   = max(len(train_ds) // train_loader.batch_size, 1)
            print(f"  {step}/{epoch_len}  train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        scheduler.step()

        # ── Validation ────────────────────────────────────────────────────────
        if (epoch + 1) % VAL_INTERVAL == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs  = val_data["img"].to(device)
                    val_labels  = val_data["seg"].to(device)

                    # Sliding window over full volume with training patch size
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

                metric = dice_metric.aggregate().item()
                dice_metric.reset()
                metric_values.append(metric)

                if metric > best_metric:
                    best_metric       = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), CHECKPOINT)
                    print("  ✓ saved new best model")

                print(
                    f"  val dice: {metric:.4f}  |  best: {best_metric:.4f} @ epoch {best_metric_epoch}"
                )
                writer.add_scalar("val_mean_dice", metric, epoch + 1)
                writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch + 1)

    print(f"\nTraining complete. Best Dice: {best_metric:.4f} at epoch {best_metric_epoch}")
    writer.close()


if __name__ == "__main__":
    main()
