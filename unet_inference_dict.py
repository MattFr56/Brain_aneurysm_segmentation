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
    Activationsd,
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

# ── Paths ──────────────────────────────────────────────────────────────────────
IMAGE_DIR  = "/kaggle/input/datasets/lorfr56/cropped-brain-vessels/inference_data"
OUTPUT_DIR = "/kaggle/working/predictions"
CHECKPOINT = "/kaggle/input/datasets/lorfr56/cropped-brain-vessels/best_metric_model.pth"
SPATIAL_SIZE = (96, 96, 16)


def main():
    print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    images = sorted(glob(os.path.join(IMAGE_DIR, "*_0000.nii*")))
    files  = [{"img": img} for img in images]
    print(f"Found {len(files)} volumes for inference")

    # ── Pre-transforms (identical to training) ─────────────────────────────────
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
    # Invertd undoes Spacingd + Orientationd so the saved mask aligns with
    # the original volume geometry (same voxel grid as the input .nii file)
    post_transforms = Compose([
        Activationsd(keys="pred", sigmoid=True),
        Invertd(
            keys="pred",
            transform=pre_transforms,
            orig_keys="img",
            nearest_interp=True,   # nearest for binary mask inversion
            to_tensor=True,
        ),
        AsDiscreted(keys="pred", threshold=0.5),
        SaveImaged(
            keys="pred",
            output_dir=OUTPUT_DIR,
            output_postfix="seg",
            resample=False,        # already in original space after Invertd
            separate_folder=False,
        ),
    ])

    # ── DataLoader ─────────────────────────────────────────────────────────────
    dataset    = Dataset(data=files, transform=pre_transforms)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=2)

    # ── Model ──────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionUnet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
    ).to(device)

    model.load_state_dict(torch.load(CHECKPOINT, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded checkpoint: {CHECKPOINT}")

    # ── Inference ──────────────────────────────────────────────────────────────
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            img = batch["img"].to(device)

            batch["pred"] = sliding_window_inference(
                inputs=img,
                roi_size=SPATIAL_SIZE,      # matches training patch size
                sw_batch_size=4,
                predictor=model,
                overlap=0.25,
            )

            # Decollate → apply post_transforms → saves each volume to OUTPUT_DIR
            batch = [post_transforms(item) for item in decollate_batch(batch)]
            print(f"  [{i+1}/{len(dataloader)}] saved prediction for {os.path.basename(images[i])}")

    print(f"\nDone. Predictions saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
