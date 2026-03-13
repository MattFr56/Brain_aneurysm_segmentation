# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ── Inference settings ─────────────────────────────────────────────────────────
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
    AsDiscreted,
    Compose,
    EnsureChannelFirstd,
    Invertd,
    LoadImaged,
    Orientationd,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

IMAGE_DIR    = "/kaggle/input/datasets/lorfr56/cropped-brain-vessels/inference_data"
OUTPUT_DIR   = "/kaggle/working/predictions"

# Single model inference — set CHECKPOINT_2 = None to disable ensemble
CHECKPOINT_1 = "/kaggle/working/best_metric_model-2.pth"
CHECKPOINT_2 = None   # e.g. "/kaggle/working/best_metric_model_seed43.pth"

SW_BATCH   = 4
SW_OVERLAP = 0.25

# ══════════════════════════════════════════════════════════════════════════════

# TTA: 8 flip combinations over H, W, D axes
TTA_FLIPS = [
    [],
    [2], [3], [4],
    [2, 3], [2, 4], [3, 4],
    [2, 3, 4],
]


def load_model(checkpoint_path, device):
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    channels  = ckpt.get("channels",  (64, 128, 256, 512))
    strides   = ckpt.get("strides",   (2, 2, 2))
    threshold = ckpt.get("threshold", 0.4)

    print(f"Checkpoint: {os.path.basename(checkpoint_path)}")
    print(f"  channels  : {channels}")
    print(f"  strides   : {strides}")
    print(f"  best_dice : {ckpt.get('best_dice', '?')}")
    print(f"  best_hd95 : {ckpt.get('best_hd95', '?')}")
    print(f"  threshold : {threshold}")
    print(f"  epoch     : {ckpt.get('epoch', '?')}")

    model = AttentionUnet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=channels,
        strides=strides,
    ).to(device)

    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model, threshold, ckpt.get("spatial_size", (128, 128, 32))


def tta_predict(model, inputs, roi_size, sw_batch_size, overlap):
    """Returns mean sigmoid probability map over 8 TTA flips."""
    preds = []
    for dims in TTA_FLIPS:
        x = torch.flip(inputs, dims) if dims else inputs
        with torch.no_grad():
            pred = sliding_window_inference(
                x,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                predictor=model,
                overlap=overlap,
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

    # ── Load model(s) ──────────────────────────────────────────────────────────
    model1, threshold, spatial_size = load_model(CHECKPOINT_1, device)

    model2 = None
    if CHECKPOINT_2 is not None:
        model2, _, _ = load_model(CHECKPOINT_2, device)
        print("✓ Ensemble mode: averaging 2 models")
    else:
        print("✓ Single model mode")

    # ── Data ───────────────────────────────────────────────────────────────────
    images = sorted(glob(os.path.join(IMAGE_DIR, "*_0000.nii*")))
    files  = [{"img": img} for img in images]
    print(f"Found {len(files)} volumes for inference")

    # ── Pre-transforms (must match training) ──────────────────────────────────
    pre_transforms = Compose([
        LoadImaged(keys="img"),
        EnsureChannelFirstd(keys="img"),
        Orientationd(keys="img", axcodes="RAS"),
        Spacingd(keys="img", pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        ScaleIntensityRanged(
            keys="img",
            a_min=100, a_max=600,
            b_min=0.0, b_max=1.0,
            clip=True,
        ),
    ])

    # ── Post-transforms ────────────────────────────────────────────────────────
    post_transforms = Compose([
        Invertd(
            keys="pred",
            transform=pre_transforms,
            orig_keys="img",
            nearest_interp=True,
            to_tensor=True,
        ),
        AsDiscreted(keys="pred", threshold=0.5),
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

    # ── Inference ─────────────────────────────────────────────────────────────
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            img = batch["img"].to(device)

            # TTA predictions
            prob1 = tta_predict(model1, img, spatial_size, SW_BATCH, SW_OVERLAP)

            if model2 is not None:
                prob2 = tta_predict(model2, img, spatial_size, SW_BATCH, SW_OVERLAP)
                mean_prob = (prob1 + prob2) / 2.0
            else:
                mean_prob = prob1

            # Threshold to binary mask
            batch["pred"] = (mean_prob > threshold).float()

            batch = [post_transforms(item) for item in decollate_batch(batch)]
            print(f"  [{i+1}/{len(dataloader)}] {os.path.basename(images[i])} → saved")

    print(f"\n✓ Done. Predictions saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
