import logging
import os
import sys
from glob import glob

import torch
from monai.config import print_config
from monai.data import Dataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.networks.nets import AttentionUnet
from monai.transforms import (
    AsDiscreted, Compose, EnsureChannelFirstd, Invertd,
    LoadImaged, Orientationd, SaveImaged,
    ScaleIntensityRanged, Spacingd,
)
# ── Config ─────────────────────────────────────────────────────────────────────
IMAGE_DIR    = "/kaggle/input/datasets/mattfr56/ensemble-pth/inference_volume"
OUTPUT_DIR   = "/kaggle/working/output"

CHECKPOINT_0 = "/kaggle/input/datasets/mattfr56/ensemble-pth/best_metric_model_3.pth"
CHECKPOINT_1 = "/kaggle/input/datasets/mattfr56/ensemble-pth/best_model.pth"
CHECKPOINT_2 = "/kaggle/input/datasets/mattfr56/ensemble-pth/best_model_phase1.pth"
CHECKPOINT_3 = "/kaggle/input/datasets/mattfr56/ensemble-pth/best_model_phase1b.pth"
CHECKPOINT_4 = "/kaggle/input/datasets/mattfr56/ensemble-pth/best_model_phase1b2.pth"

# ── Weights — equal to start, tune based on individual val Dice ────────────────
# To weight by performance, set to val Dice scores:
# e.g. WEIGHT_0=0.82, WEIGHT_1=0.79, WEIGHT_2=0.78 etc.
WEIGHT_0 = 1.0
WEIGHT_1 = 1.0
WEIGHT_2 = 1.0
WEIGHT_3 = 1.0
WEIGHT_4 = 1.0

SW_BATCH   = 2      # safe for T4 VRAM with 5 models
SW_OVERLAP = 0.5

# ── TTA — 16 augmentations (8 flips x 2 rotations) ───────────────────────────
TTA_FLIPS     = [[], [2], [3], [4], [2,3], [2,4], [3,4], [2,3,4]]
TTA_ROTATIONS = [0, 2]  # 0° and 180°


def load_model(checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt      = torch.load(checkpoint_path, map_location=device,
                           weights_only=False)
    channels  = ckpt.get("channels",     (64, 128, 256, 512))
    strides   = ckpt.get("strides",      (2, 2, 2))
    threshold = ckpt.get("threshold",    0.4)
    hu_min    = ckpt.get("hu_min",       100)
    hu_max    = ckpt.get("hu_max",       400)
    spatial   = ckpt.get("spatial_size", (128, 128, 32))

    print(f"Checkpoint : {os.path.basename(checkpoint_path)}")
    print(f"  channels : {channels} | strides: {strides}")
    print(f"  best_dice: {ckpt.get('best_dice', '?'):.4f}")
    print(f"  best_hd95: {ckpt.get('best_hd95', '?'):.2f}mm")
    print(f"  threshold: {threshold} | hu: [{hu_min},{hu_max}]")
    print(f"  epoch    : {ckpt.get('epoch', '?')}")

    model = AttentionUnet(
        spatial_dims=3, in_channels=1, out_channels=1,
        channels=channels, strides=strides,
    ).to(device)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model, threshold, spatial, hu_min, hu_max


def tta_predict(model, inputs, roi_size, sw_batch_size, overlap):
    """16 augmentations — 8 flips x 2 rotations."""
    preds = []
    for axes in TTA_FLIPS:
        for k in TTA_ROTATIONS:
            x = torch.flip(inputs, axes) if axes else inputs.clone()
            if k > 0:
                x = torch.rot90(x, k, dims=[2, 3])
            with torch.no_grad():
                pred = sliding_window_inference(
                    x, roi_size=roi_size,
                    sw_batch_size=sw_batch_size,
                    predictor=model, overlap=overlap,
                )
            if k > 0:
                pred = torch.rot90(pred, -k, dims=[2, 3])
            if axes:
                pred = torch.flip(pred, axes)
            preds.append(torch.sigmoid(pred))
    return torch.stack(preds).mean(dim=0)


def main():
    print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load all 5 models ──────────────────────────────────────────────────────
    print("\n=== Loading ensemble models ===")
    model0, threshold, spatial_size, hu_min, hu_max = load_model(
        CHECKPOINT_0, device)
    model1, _, _, _, _ = load_model(CHECKPOINT_1, device)
    model2, _, _, _, _ = load_model(CHECKPOINT_2, device)
    model3, _, _, _, _ = load_model(CHECKPOINT_3, device)
    model4, _, _, _, _ = load_model(CHECKPOINT_4, device)

    # Normalize weights to sum to 1
    raw_weights = [WEIGHT_0, WEIGHT_1, WEIGHT_2, WEIGHT_3, WEIGHT_4]
    total       = sum(raw_weights)
    weights     = [w / total for w in raw_weights]
    models      = [model0, model1, model2, model3, model4]
    checkpoints = [CHECKPOINT_0, CHECKPOINT_1, CHECKPOINT_2,
                   CHECKPOINT_3, CHECKPOINT_4]

    print(f"\n✓ Ensemble: {len(models)} models")
    for ckpt, w in zip(checkpoints, weights):
        print(f"  {os.path.basename(ckpt):40s} weight={w:.3f}")
    print(f"  Threshold : {threshold:.2f}")
    print(f"  HU window : [{hu_min}, {hu_max}]")
    print(f"  Patch size: {spatial_size}")
    print(f"  TTA augs  : {len(TTA_FLIPS) * len(TTA_ROTATIONS)}")

    # ── Data ───────────────────────────────────────────────────────────────────
    images = sorted(glob(os.path.join(IMAGE_DIR, "*.nii*")))
    files  = [{"img": img} for img in images]
    print(f"\nFound {len(files)} volumes for inference")

    # ── Transforms — must match training exactly ───────────────────────────────
    pre_transforms = Compose([
        LoadImaged(keys="img"),
        EnsureChannelFirstd(keys="img"),
        Orientationd(keys="img", axcodes="RAS"),
        Spacingd(keys="img", pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        ScaleIntensityRanged(keys="img",
                             a_min=hu_min, a_max=hu_max,
                             b_min=0.0, b_max=1.0, clip=True),
    ])

    post_transforms = Compose([
        Invertd(
            keys="pred",
            transform=pre_transforms,
            orig_keys="img",
            nearest_interp=True,
            to_tensor=True,
        ),
        AsDiscreted(keys="pred", threshold=threshold),
        SaveImaged(
            keys="pred",
            output_dir=OUTPUT_DIR,
            output_postfix="seg",
            resample=False,
            separate_folder=False,
        ),
    ])

    dataset    = Dataset(data=files, transform=pre_transforms)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=2)

    # ── Inference ──────────────────────────────────────────────────────────────
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            fname = os.path.basename(images[i])
            print(f"\n[{i+1}/{len(dataloader)}] {fname}")

            img       = batch["img"].to(device)
            mean_prob = torch.zeros(1, 1, *img.shape[2:]).to(device)

            # Weighted ensemble — all 5 models
            for j, (model, w) in enumerate(zip(models, weights)):
                print(f"  Model {j+1}/{len(models)} "
                      f"(weight={w:.3f})...")
                prob       = tta_predict(model, img, spatial_size,
                                         SW_BATCH, SW_OVERLAP)
                mean_prob += w * prob.to(device)

            # Threshold to binary — no post-processing
            batch["pred"] = (mean_prob > threshold).float().cpu()
            batch = [post_transforms(item)
                     for item in decollate_batch(batch)]
            print(f"  -> saved to {OUTPUT_DIR}")

    print(f"\n✓ Done — {len(files)} volumes processed")
    print(f"  Ensemble : {len(models)} models")
    print(f"  Weights  : {[f'{w:.3f}' for w in weights]}")
    print(f"  Output   -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
