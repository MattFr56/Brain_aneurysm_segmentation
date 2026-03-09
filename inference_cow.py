"""
CoW Segmentation — 2.5D Inference Script
==========================================
- Input  : folder of NIfTI CTA volumes (.nii or .nii.gz)
- Output : folder of predicted binary masks (.nii.gz)
- Model  : trained AttentionUnet 2.5D checkpoint
           (3-channel input: slice s-1, s, s+1)

Features:
  - TTA: 4 augmentations (original + hflip + vflip + both) averaged
  - Multi-scale: 0.75×, 1.0×, 1.25× averaged (--multi_scale flag)
  - Batch inference for speed
  - Saves mask with same affine/header as input

Supports two filename patterns:
  - topcow_ct_XXX_0000.nii   (TopCoW format)
  - Volume_XXXXX.nii(.gz)    (custom format)

Usage in Colab:
    python inference_cow_25d.py \
        -input_dir  /content/drive/MyDrive/test_images \
        -output_dir /content/drive/MyDrive/predicted_masks \
        -checkpoint /content/drive/MyDrive/CoW_checkpoints/model_best_AttUNet25D_CoW.pth.tar \
        --multi_scale --tta
"""

import argparse
import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
import cv2
from tqdm import tqdm
from monai.networks.nets import AttentionUnet

# ─── ARGS ─────────────────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser(description="CoW 2.5D Segmentation Inference")
    parser.add_argument("-input_dir",  required=True,
                        help="folder containing CTA volumes (.nii or .nii.gz)")
    parser.add_argument("-output_dir", required=True,
                        help="folder to save predicted masks (.nii.gz)")
    parser.add_argument("-checkpoint", required=True,
                        help="path to trained 2.5D model checkpoint (.pth.tar)")
    parser.add_argument("-img_size",   default=256,  type=int)
    parser.add_argument("-threshold",  default=0.5,  type=float,
                        help="sigmoid threshold for binarization (default 0.5)")
    parser.add_argument("-hu_min",     default=100,  type=int)
    parser.add_argument("-hu_max",     default=500,  type=int)
    parser.add_argument("--tta",       action="store_true",
                        help="enable test-time augmentation (4 flips averaged)")
    parser.add_argument("--multi_scale", action="store_true",
                        help="enable multi-scale inference (0.75, 1.0, 1.25)")
    parser.add_argument("--gpu",       default=0,    type=int)
    return parser.parse_args()

# ─── LOAD MODEL ───────────────────────────────────────────────────────────────

def load_model(checkpoint_path, device):
    model = AttentionUnet(
        spatial_dims=2,
        in_channels=3,                      # 2.5D: (s-1, s, s+1)
        out_channels=1,
        channels=(64, 128, 256, 512, 1024), # must match training architecture
        strides=(2, 2, 2, 2),
        dropout=0.2,
    ).to(device)

    print(f"Loading checkpoint from {checkpoint_path} ...")
    ckpt       = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)

    # Strip DataParallel prefix if present
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # Remap MONAI version differences (sub0→submodule.0 etc.)
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

# ─── PREPROCESS VOLUME ────────────────────────────────────────────────────────

def preprocess_volume(img_vol, img_size, hu_min, hu_max):
    """
    HU windowing + resize all slices.
    Returns: (img_size, img_size, D) float32 array
    """
    H, W, D = img_vol.shape
    out = np.zeros((img_size, img_size, D), dtype=np.float32)
    for s in range(D):
        sl = np.clip(img_vol[:, :, s], hu_min, hu_max)
        sl = (sl - hu_min) / (hu_max - hu_min)
        out[:, :, s] = cv2.resize(sl.astype(np.float32), (img_size, img_size),
                                   interpolation=cv2.INTER_LINEAR)
    return out

# ─── SINGLE SCALE PREDICTION ──────────────────────────────────────────────────

def predict_single(model, inp, device):
    """inp: (1, 3, H, W) tensor — returns (1, 1, H, W) sigmoid prob"""
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            return torch.sigmoid(model(inp))

# ─── TTA PREDICTION ───────────────────────────────────────────────────────────

def predict_tta(model, inp, device):
    """
    4-fold TTA: original + hflip + vflip + both flips, averaged.
    inp: (1, 3, H, W)
    """
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            p0 = torch.sigmoid(model(inp))
            p1 = torch.sigmoid(model(torch.flip(inp, dims=[-1])))
            p1 = torch.flip(p1, dims=[-1])
            p2 = torch.sigmoid(model(torch.flip(inp, dims=[-2])))
            p2 = torch.flip(p2, dims=[-2])
            p3 = torch.sigmoid(model(torch.flip(inp, dims=[-1, -2])))
            p3 = torch.flip(p3, dims=[-1, -2])
    return (p0 + p1 + p2 + p3) / 4.0

# ─── MULTI-SCALE PREDICTION ───────────────────────────────────────────────────

def predict_multiscale(model, inp, device, scales=(0.75, 1.0, 1.25),
                       use_tta=False):
    """
    Run inference at multiple scales and average predictions.
    inp: (1, 3, H, W)
    """
    H, W   = inp.shape[-2], inp.shape[-1]
    preds  = []
    for s in scales:
        if s != 1.0:
            new_H, new_W = int(H * s), int(W * s)
            inp_s = F.interpolate(inp, size=(new_H, new_W),
                                  mode='bilinear', align_corners=False)
        else:
            inp_s = inp
        if use_tta:
            pred_s = predict_tta(model, inp_s, device)
        else:
            pred_s = predict_single(model, inp_s, device)
        if s != 1.0:
            pred_s = F.interpolate(pred_s, size=(H, W),
                                   mode='bilinear', align_corners=False)
        preds.append(pred_s)
    return torch.stack(preds).mean(0)

# ─── INFERENCE ON ONE VOLUME ──────────────────────────────────────────────────

def inference_volume(model, img_vol, args, device):
    """
    Run 2.5D slice-by-slice inference on a 3D volume.
    Each slice uses triplet (s-1, s, s+1) as input channels.
    First and last slices use edge padding (replicate border).
    Returns predicted mask with same H×W×D as input.
    """
    H_orig, W_orig, D = img_vol.shape

    # Preprocess all slices at once
    vol_pre = preprocess_volume(img_vol, args.img_size, args.hu_min, args.hu_max)
    # (img_size, img_size, D)

    mask_pre = np.zeros((args.img_size, args.img_size, D), dtype=np.float32)

    for s in tqdm(range(D), desc="  Slices", leave=False):
        # Build triplet with edge padding
        s_prev = max(0, s - 1)
        s_next = min(D - 1, s + 1)

        triplet = np.stack([
            vol_pre[:, :, s_prev],
            vol_pre[:, :, s],
            vol_pre[:, :, s_next],
        ], axis=0)  # (3, H, W)

        inp = torch.from_numpy(triplet).unsqueeze(0).to(device)  # (1,3,H,W)

        # Predict
        if args.multi_scale:
            pred = predict_multiscale(model, inp, device,
                                      scales=(0.75, 1.0, 1.25),
                                      use_tta=args.tta)
        elif args.tta:
            pred = predict_tta(model, inp, device)
        else:
            pred = predict_single(model, inp, device)

        mask_pre[:, :, s] = pred[0, 0].cpu().float().numpy()

    # Resize mask back to original H×W
    mask_vol = np.zeros((H_orig, W_orig, D), dtype=np.float32)
    for s in range(D):
        mask_vol[:, :, s] = cv2.resize(
            mask_pre[:, :, s], (W_orig, H_orig),
            interpolation=cv2.INTER_LINEAR)

    # Binarize
    mask_vol = (mask_vol > args.threshold).astype(np.uint8)
    return mask_vol

# ─── FIND ALL VOLUMES ─────────────────────────────────────────────────────────

def find_volumes(input_dir):
    """Find all NIfTI files, deduplicated."""
    files  = sorted(glob.glob(os.path.join(input_dir, "*.nii.gz")))
    files += sorted(glob.glob(os.path.join(input_dir, "*.nii")))
    seen, clean = set(), []
    for f in files:
        base = f.replace(".nii.gz", "").replace(".nii", "")
        if base not in seen:
            seen.add(base)
            clean.append(f)
    return sorted(clean)

# ─── OUTPUT FILENAME ──────────────────────────────────────────────────────────

def get_output_name(input_path):
    basename = os.path.basename(input_path)
    name     = basename.replace(".nii.gz", "").replace(".nii", "")
    name     = name.replace("_0000", "")
    return f"{name}_mask.nii.gz"

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu}"
                          if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"TTA: {'✅' if args.tta else '❌'}  |  "
          f"Multi-scale: {'✅' if args.multi_scale else '❌'}")

    model = load_model(args.checkpoint, device)

    volumes = find_volumes(args.input_dir)
    print(f"Found {len(volumes)} volumes in {args.input_dir}")
    if len(volumes) == 0:
        print("❌ No NIfTI files found — check input_dir path")
        return

    for img_path in tqdm(volumes, desc="Volumes"):
        nii     = nib.load(img_path)
        img_vol = nii.get_fdata().astype(np.float32)
        affine  = nii.affine
        header  = nii.header

        mask_vol = inference_volume(model, img_vol, args, device)

        out_name = get_output_name(img_path)
        out_path = os.path.join(args.output_dir, out_name)
        nib.save(nib.Nifti1Image(mask_vol, affine, header), out_path)
        print(f"  ✅ {out_name}")

    print(f"\n✅ Done — {len(volumes)} masks saved to {args.output_dir}")

if __name__ == "__main__":
    main()
