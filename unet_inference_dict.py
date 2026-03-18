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
IMAGE_DIR    = "/content/data"
OUTPUT_DIR   = "/content/output"
CHECKPOINT_1 = "/content/best_model_phase1.pth"
CHECKPOINT_2 = None  # optional second model for ensemble
SW_BATCH     = 8
SW_OVERLAP   = 0.5   # matches val overlap for consistent results

TTA_FLIPS = [[], [2], [3], [4], [2,3], [2,4], [3,4], [2,3,4]]


def load_model(checkpoint_path, device):
    ckpt      = torch.load(checkpoint_path, map_location=device, weights_only=False)
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
    preds = []
    for dims in TTA_FLIPS:
        x = torch.flip(inputs, dims) if dims else inputs
        with torch.no_grad():
            pred = sliding_window_inference(
                x, roi_size=roi_size, sw_batch_size=sw_batch_size,
                predictor=model, overlap=overlap,
            )
        if dims:
            pred = torch.flip(pred, dims)
        preds.append(torch.sigmoid(pred))
    return torch.stack(preds).mean(dim=0)


def main():
    print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load models ────────────────────────────────────────────────────────────
    model1, threshold, spatial_size, hu_min, hu_max = load_model(
        CHECKPOINT_1, device
    )
    model2 = None
    if CHECKPOINT_2 is not None:
        model2, _, _, _, _ = load_model(CHECKPOINT_2, device)
        print("✓ Ensemble mode: averaging 2 models")
    else:
        print("✓ Single model mode")

    # ── Data ───────────────────────────────────────────────────────────────────
    images = sorted(glob(os.path.join(IMAGE_DIR, "*.nii*")))
    files  = [{"img": img} for img in images]
    print(f"Found {len(files)} volumes for inference")

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
            img   = batch["img"].to(device)
            prob1 = tta_predict(model1, img, spatial_size, SW_BATCH, SW_OVERLAP)

            if model2 is not None:
                prob2     = tta_predict(model2, img, spatial_size, SW_BATCH, SW_OVERLAP)
                mean_prob = (prob1 + prob2) / 2.0
            else:
                mean_prob = prob1

            batch["pred"] = (mean_prob > threshold).float()
            batch = [post_transforms(item) for item in decollate_batch(batch)]
            print(f"  [{i+1}/{len(dataloader)}] "
                  f"{os.path.basename(images[i])} → saved")

    print(f"\n✓ Done — predictions saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
