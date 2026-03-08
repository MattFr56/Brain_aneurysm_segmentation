"""
CoW Segmentation — Inference Script
=====================================
- Input  : folder of NIfTI CTA volumes (.nii or .nii.gz)
- Output : folder of predicted binary masks (.nii.gz)
- Model  : trained AttentionUnet 2D checkpoint

Supports two filename patterns:
  - topcow_ct_XXX_0000.nii   (TopCoW format)
  - Volume_XXXXX.nii(.gz)    (custom format)

Usage in Colab:
    python inference_cow.py \
        -input_dir  /content/drive/MyDrive/test_images \
        -output_dir /content/drive/MyDrive/predicted_masks \
        -checkpoint /content/drive/MyDrive/CoW_checkpoints/model_best_AttUNet_CoW.pth.tar \
        -img_size 256
"""

import argparse
import os
import glob
import numpy as np
import torch
import nibabel as nib
import cv2
from tqdm import tqdm
from monai.networks.nets import AttentionUnet

# ─── ARGS ─────────────────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser(description="CoW Segmentation Inference")
    parser.add_argument("-input_dir",  required=True,
                        help="folder containing CTA volumes (.nii or .nii.gz)")
    parser.add_argument("-output_dir", required=True,
                        help="folder to save predicted masks (.nii.gz)")
    parser.add_argument("-checkpoint", required=True,
                        help="path to trained model checkpoint (.pth.tar)")
    parser.add_argument("-img_size",   default=256, type=int,
                        help="image size used during training (default 256)")
    parser.add_argument("-threshold",  default=0.5, type=float,
                        help="sigmoid threshold for binarization (default 0.5)")
    parser.add_argument("-hu_min",     default=100, type=int)
    parser.add_argument("-hu_max",     default=500, type=int)
    parser.add_argument("--gpu",       default=0,   type=int)
    return parser.parse_args()

# ─── LOAD MODEL ───────────────────────────────────────────────────────────────

def load_model(checkpoint_path, device):
    model = AttentionUnet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
    ).to(device)

    print(f"Loading checkpoint from {checkpoint_path} ...")
    ckpt       = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # Remap MONAI version differences:
    # older MONAI: sub0/sub1/subconv
    # newer MONAI: submodule.0/submodule.1/submodule.conv
    remapped = {}
    for k, v in state_dict.items():
        k = k.replace(".sub0.", ".submodule.0.")
        k = k.replace(".sub1.", ".submodule.1.")
        k = k.replace(".subconv.", ".submodule.conv.")
        remapped[k] = v
    state_dict = remapped

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"✅ Model loaded")
    return model

# ─── PREPROCESS SLICE ─────────────────────────────────────────────────────────

def preprocess_slice(sl, img_size, hu_min, hu_max):
    sl = np.clip(sl, hu_min, hu_max)
    sl = (sl - hu_min) / (hu_max - hu_min)
    sl = cv2.resize(sl.astype(np.float32), (img_size, img_size),
                    interpolation=cv2.INTER_LINEAR)
    return sl

# ─── INFERENCE ON ONE VOLUME ──────────────────────────────────────────────────

def inference_volume(model, img_vol, args, device):
    """
    Run slice-by-slice inference on a 3D volume.
    Returns predicted mask with same spatial dimensions as input.
    """
    H_orig, W_orig, n_slices = img_vol.shape
    mask_vol = np.zeros((H_orig, W_orig, n_slices), dtype=np.float32)

    with torch.no_grad():
        for s in range(n_slices):
            sl = img_vol[:, :, s].astype(np.float32)

            # Preprocess
            sl_pre = preprocess_slice(sl, args.img_size, args.hu_min, args.hu_max)

            # To tensor
            inp = torch.from_numpy(sl_pre).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W)

            # Predict
            pred = torch.sigmoid(model(inp))
            pred = pred[0, 0].cpu().numpy()  # (H,W)

            # Resize back to original slice dimensions
            pred_orig = cv2.resize(pred, (W_orig, H_orig),
                                   interpolation=cv2.INTER_LINEAR)

            # Binarize
            mask_vol[:, :, s] = (pred_orig > args.threshold).astype(np.float32)

    return mask_vol

# ─── FIND ALL VOLUMES ─────────────────────────────────────────────────────────

def find_volumes(input_dir):
    """Find all NIfTI files, supporting both .nii and .nii.gz"""
    files  = sorted(glob.glob(os.path.join(input_dir, "*.nii.gz")))
    files += sorted(glob.glob(os.path.join(input_dir, "*.nii")))
    # Remove duplicates (in case .nii.gz and .nii both match)
    seen  = set()
    clean = []
    for f in files:
        base = f.replace(".nii.gz", "").replace(".nii", "")
        if base not in seen:
            seen.add(base)
            clean.append(f)
    return sorted(clean)

# ─── OUTPUT FILENAME ──────────────────────────────────────────────────────────

def get_output_name(input_path):
    """
    Generate output mask filename from input filename.
    topcow_ct_001_0000.nii   → topcow_ct_001_mask.nii.gz
    Volume_00001.nii.gz      → Volume_00001_mask.nii.gz
    """
    basename = os.path.basename(input_path)
    # Strip extensions
    name = basename.replace(".nii.gz", "").replace(".nii", "")
    # Strip _0000 suffix if present (TopCoW format)
    name = name.replace("_0000", "")
    return f"{name}_mask.nii.gz"

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu}"
                          if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = load_model(args.checkpoint, device)

    # Find volumes
    volumes = find_volumes(args.input_dir)
    print(f"Found {len(volumes)} volumes in {args.input_dir}")

    if len(volumes) == 0:
        print("❌ No NIfTI files found — check input_dir path")
        return

    # Run inference
    for img_path in tqdm(volumes, desc="Inference"):
        # Load volume
        nii     = nib.load(img_path)
        img_vol = nii.get_fdata().astype(np.float32)
        affine  = nii.affine
        header  = nii.header

        # Inference
        mask_vol = inference_volume(model, img_vol, args, device)

        # Save mask with same affine as input
        out_name = get_output_name(img_path)
        out_path = os.path.join(args.output_dir, out_name)
        mask_nii = nib.Nifti1Image(mask_vol.astype(np.uint8), affine, header)
        nib.save(mask_nii, out_path)

    print(f"\n✅ Done — {len(volumes)} masks saved to {args.output_dir}")

if __name__ == "__main__":
    main()
