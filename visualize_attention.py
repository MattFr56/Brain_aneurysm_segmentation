import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from monai.networks.nets import AttentionUnet
from monai.visualize import GradCAM
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    Orientationd, Spacingd, ScaleIntensityRanged,
    ToTensord,
)
from monai.inferers import sliding_window_inference

# ── Config ─────────────────────────────────────────────────────────────────────
IMAGE_PATH   = "/content/data/CT.Seq5.Ser7.Img1.nii.gz"  # ← your CTA volume
CHECKPOINT   = "/content/best_model_phase1b.pth"    # ← your checkpoint
OUTPUT_DIR   = "/content/gradcam"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SW_BATCH     = 2
SW_OVERLAP   = 0.5
SPATIAL_SIZE = (128, 128, 32)

# ── Slices to visualize ────────────────────────────────────────────────────────
# Set to your specific z-slice indices (original volume coordinates)
# Set to None for automatic selection (slices with most foreground)
ORIGINAL_SLICES = [416, 459, 463, 465, 508]
ORIGINAL_SPACING = 0.5
TARGET_SPACING = 1.0
SPECIFIC_SLICES = [int(s * ORIGINAL_SPACING/TARGET_SPACING) for s in ORIGINAL_SLICES]
print(f"Converted slices: {SPECIFIC_SLICES}")# ← your slices here
N_SLICES        = 8   # only used if SPECIFIC_SLICES = None

# ── Target layers for GradCAM ──────────────────────────────────────────────────
TARGET_LAYERS = [
    "model.1.merge",                              # decoder level 1 (shallow)
    "model.1.submodule.1.merge",                  # decoder level 2
    "model.1.submodule.1.submodule.1.merge",      # decoder level 3 (deep)
]


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

    model = AttentionUnet(
        spatial_dims=3, in_channels=1, out_channels=1,
        channels=channels, strides=strides,
    ).to(device)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded: {os.path.basename(checkpoint_path)}")
    print(f"  channels={channels} | threshold={threshold} | "
          f"hu=[{hu_min},{hu_max}]")
    return model, threshold, spatial, hu_min, hu_max


def run_gradcam_sliding_window(model, img, spatial_size,
                                target_layer, device,
                                overlap=0.25):
    """
    Run GradCAM over sliding window patches and stitch into full volume.
    Returns GradCAM map same shape as img (H, W, D).
    """
    _, _, H, W, D = img.shape
    gradcam_vol   = np.zeros((H, W, D), dtype=np.float32)
    count_vol     = np.zeros((H, W, D), dtype=np.float32)

    stride = [max(1, int(s * (1 - overlap))) for s in spatial_size]
    cam    = GradCAM(nn_module=model, target_layers=target_layer)

    positions = [
        (h, w, d)
        for h in range(0, max(1, H - spatial_size[0] + 1), stride[0])
        for w in range(0, max(1, W - spatial_size[1] + 1), stride[1])
        for d in range(0, max(1, D - spatial_size[2] + 1), stride[2])
    ]
    total = len(positions)

    for count, (h, w, d) in enumerate(positions, 1):
        if count % 10 == 0:
            print(f"    patch {count}/{total}...", end="\r")

        h2 = min(h + spatial_size[0], H)
        w2 = min(w + spatial_size[1], W)
        d2 = min(d + spatial_size[2], D)

        patch = img[:, :, h:h2, w:w2, d:d2].clone()

        # Pad if needed
        ph = spatial_size[0] - patch.shape[2]
        pw = spatial_size[1] - patch.shape[3]
        pd = spatial_size[2] - patch.shape[4]
        if ph > 0 or pw > 0 or pd > 0:
            patch = torch.nn.functional.pad(patch, (0, pd, 0, pw, 0, ph))

        try:
            patch_req = patch.requires_grad_(True)
            result    = cam(x=patch_req)
            gc        = result[0, 0].detach().cpu().numpy()
            gc        = gc[:h2-h, :w2-w, :d2-d]
            gradcam_vol[h:h2, w:w2, d:d2] += gc
            count_vol[h:h2, w:w2, d:d2]   += 1
        except Exception:
            pass

    count_vol   = np.maximum(count_vol, 1)
    gradcam_vol /= count_vol
    print()
    return gradcam_vol


def get_slices(pred_np, specific_slices, n_slices, vol_depth):
    """Return slice indices to visualize."""
    if specific_slices is not None:
        slices = [max(0, min(s, vol_depth - 1)) for s in specific_slices]
        print(f"Using specific slices: {slices}")
    else:
        fg_per_slice = (pred_np > 0).sum(axis=(0, 1))
        if fg_per_slice.max() > 0:
            slices = sorted(
                np.argsort(fg_per_slice)[-n_slices:].tolist()
            )
        else:
            slices = np.linspace(0, vol_depth-1, n_slices, dtype=int).tolist()
        print(f"Auto-selected slices: {slices}")
    return slices


def normalize_map(arr):
    """Normalize array to [0, 1]."""
    mn, mx = arr.min(), arr.max()
    if mx > mn:
        return (arr - mn) / (mx - mn)
    return arr


def resize_to_match(arr, target_shape):
    """Resize arr to target_shape using zoom."""
    if arr.shape == target_shape:
        return arr
    scale = [target_shape[i] / arr.shape[i] for i in range(3)]
    return zoom(arr, scale, order=1)


def visualize_gradcam(img_np, pred_np, gradcam_maps,
                       output_dir, specific_slices=None, n_slices=8):
    """
    Plot GradCAM overlays for each target layer.
    img_np, pred_np: (H, W, D)
    gradcam_maps: dict {layer_name: (H, W, D)}
    """
    slices   = get_slices(pred_np, specific_slices,
                           n_slices, img_np.shape[2])
    n_layers = len(gradcam_maps)
    cols     = 2 + n_layers

    # Preprocess GradCAM maps
    gc_processed = {}
    for layer_name, gc_map in gradcam_maps.items():
        gc_map = resize_to_match(gc_map, img_np.shape)
        gc_map = normalize_map(gc_map)
        gc_processed[layer_name] = gc_map

    # ── Per-slice plots ────────────────────────────────────────────────────────
    for slice_idx in slices:
        fig, axes = plt.subplots(1, cols, figsize=(cols * 4, 4))
        fig.suptitle(f"GradCAM — Slice z={slice_idx}",
                     fontsize=12, fontweight="bold")

        # Image
        axes[0].imshow(img_np[:, :, slice_idx].T,
                       cmap="gray", origin="lower", vmin=0, vmax=1)
        axes[0].set_title("Image", fontsize=10)
        axes[0].axis("off")

        # Prediction overlay
        axes[1].imshow(img_np[:, :, slice_idx].T,
                       cmap="gray", origin="lower", vmin=0, vmax=1)
        axes[1].imshow(pred_np[:, :, slice_idx].T,
                       cmap="hot", alpha=0.5, origin="lower")
        n_fg = int((pred_np[:, :, slice_idx] > 0).sum())
        axes[1].set_title(f"Prediction\n(fg={n_fg}vox)", fontsize=10)
        axes[1].axis("off")

        # GradCAM per layer
        for k, (layer_name, gc_map) in enumerate(gc_processed.items()):
            ax  = axes[2 + k]
            ax.imshow(img_np[:, :, slice_idx].T,
                      cmap="gray", origin="lower", vmin=0, vmax=1)
            im = ax.imshow(gc_map[:, :, slice_idx].T,
                           cmap="jet", alpha=0.6,
                           origin="lower", vmin=0, vmax=1)
            parts = layer_name.split(".")
            label = f"Layer {k+1} ({'.'.join(parts[-2:])})"
            ax.set_title(label, fontsize=8)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        out_path = os.path.join(output_dir, f"gradcam_z{slice_idx:03d}.png")
        plt.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"  Saved -> {out_path}")

    # ── MIP summary ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, cols, figsize=(cols * 4, 4))
    fig.suptitle("GradCAM — Max Intensity Projection",
                 fontsize=12, fontweight="bold")

    axes[0].imshow(img_np.max(axis=2).T, cmap="gray", origin="lower")
    axes[0].set_title("Image MIP"); axes[0].axis("off")

    axes[1].imshow(img_np.max(axis=2).T, cmap="gray", origin="lower")
    axes[1].imshow(pred_np.max(axis=2).T, cmap="hot",
                   alpha=0.5, origin="lower")
    axes[1].set_title("Prediction MIP"); axes[1].axis("off")

    for k, (layer_name, gc_map) in enumerate(gc_processed.items()):
        ax = axes[2 + k]
        ax.imshow(img_np.max(axis=2).T, cmap="gray", origin="lower")
        ax.imshow(gc_map.max(axis=2).T, cmap="jet",
                  alpha=0.6, origin="lower", vmin=0, vmax=1)
        parts = layer_name.split(".")
        ax.set_title(f"GradCAM MIP\n({'.'.join(parts[-2:])})",
                     fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "gradcam_mip.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"MIP saved -> {out_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load model ─────────────────────────────────────────────────────────────
    model, threshold, spatial_size, hu_min, hu_max = load_model(
        CHECKPOINT, device
    )

    # ── Load and preprocess image ──────────────────────────────────────────────
    transforms = Compose([
        LoadImaged(keys="img"),
        EnsureChannelFirstd(keys="img"),
        Orientationd(keys="img", axcodes="RAS"),
        Spacingd(keys="img", pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        ScaleIntensityRanged(keys="img",
                             a_min=hu_min, a_max=hu_max,
                             b_min=0.0, b_max=1.0, clip=True),
        ToTensord(keys="img"),
    ])

    data = transforms({"img": IMAGE_PATH})
    img  = data["img"].unsqueeze(0).to(device)
    print(f"\nImage shape after preprocessing: {img.shape}")

    # ── Get prediction ─────────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        output = sliding_window_inference(
            img, roi_size=spatial_size,
            sw_batch_size=SW_BATCH,
            predictor=model, overlap=SW_OVERLAP,
        )
    pred    = (torch.sigmoid(output) > threshold).float()
    pred_np = pred[0, 0].cpu().numpy()
    img_np  = img[0, 0].cpu().numpy()
    print(f"Prediction: {pred_np.sum():.0f} foreground voxels")
    print(f"Volume z-depth: {img_np.shape[2]} slices")

    # ── Compute GradCAM for each layer ─────────────────────────────────────────
    gradcam_maps = {}
    for layer_name in TARGET_LAYERS:
        print(f"\nComputing GradCAM: {layer_name}")
        gc_map = run_gradcam_sliding_window(
            model, img, spatial_size,
            target_layer=layer_name,
            device=device,
            overlap=0.25,
        )
        if gc_map is not None:
            gradcam_maps[layer_name] = gc_map
            print(f"  shape={gc_map.shape} | "
                  f"range=[{gc_map.min():.3f}, {gc_map.max():.3f}]")

    # Fallback — print available layers if none computed
    if not gradcam_maps:
        print("\nERROR: No GradCAM maps computed.")
        print("Available merge/attention layers:")
        raw = model.module if hasattr(model, "module") else model
        for name, _ in raw.named_modules():
            if "merge" in name or "attention" in name:
                print(f"  {name}")
        return

    # ── Visualize ──────────────────────────────────────────────────────────────
    print(f"\nGenerating plots...")
    visualize_gradcam(
        img_np=img_np,
        pred_np=pred_np,
        gradcam_maps=gradcam_maps,
        output_dir=OUTPUT_DIR,
        specific_slices=SPECIFIC_SLICES,
        n_slices=N_SLICES,
    )

    print(f"\n✓ Done — GradCAM saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
