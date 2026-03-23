import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage import zoom
from skimage import measure

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
# Set to your specific z-slice indices (in resampled 1mm space)
# Set to None for automatic selection (slices with most foreground)
ORIGINAL_SLICES = [416, 459, 463, 465, 508]
ORIGINAL_SPACING = 0.5
TARGET_SPACING = 1.0
SPECIFIC_SLICES = [int(s * ORIGINAL_SPACING/TARGET_SPACING) for s in ORIGINAL_SLICES]
print(f"Converted slices: {SPECIFIC_SLICES}")# ← your slices here
N_SLICES        = 8     # only used if SPECIFIC_SLICES = None

# ── GradCAM style ──────────────────────────────────────────────────────────────
HEATMAP_ALPHA   = 0.55   # transparency of heatmap over image
CONTOUR_COLOR   = "white"
CONTOUR_WIDTH   = 1.5

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
                                target_layer, device, overlap=0.25):
    _, _, H, W, D = img.shape
    gradcam_vol   = np.zeros((H, W, D), dtype=np.float32)
    count_vol     = np.zeros((H, W, D), dtype=np.float32)
    stride        = [max(1, int(s * (1 - overlap))) for s in spatial_size]
    cam           = GradCAM(nn_module=model, target_layers=target_layer)

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
        ph = spatial_size[0] - patch.shape[2]
        pw = spatial_size[1] - patch.shape[3]
        pd = spatial_size[2] - patch.shape[4]
        if ph > 0 or pw > 0 or pd > 0:
            patch = torch.nn.functional.pad(patch, (0, pd, 0, pw, 0, ph))
        try:
            result = cam(x=patch.requires_grad_(True))
            gc     = result[0, 0].detach().cpu().numpy()
            gc     = gc[:h2-h, :w2-w, :d2-d]
            gradcam_vol[h:h2, w:w2, d:d2] += gc
            count_vol[h:h2, w:w2, d:d2]   += 1
        except Exception:
            pass

    count_vol   = np.maximum(count_vol, 1)
    gradcam_vol /= count_vol
    print()
    return gradcam_vol


def normalize_map(arr):
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn) if mx > mn else arr


def resize_to_match(arr, target_shape):
    if arr.shape == target_shape:
        return arr
    scale = [target_shape[i] / arr.shape[i] for i in range(3)]
    return zoom(arr, scale, order=1)


def blend_heatmap(img_slice, gc_slice, alpha=0.55):
    """
    Blend GradCAM heatmap over grayscale image — same style as cat image.
    img_slice: (H, W) float [0,1]
    gc_slice:  (H, W) float [0,1]
    Returns:   (H, W, 3) RGB
    """
    # Convert grayscale to RGB
    img_rgb = np.stack([img_slice, img_slice, img_slice], axis=-1)

    # Apply jet colormap to GradCAM
    heatmap_rgb = cm.jet(gc_slice)[:, :, :3]  # (H, W, 3)

    # Blend
    blended = (1 - alpha) * img_rgb + alpha * heatmap_rgb
    return np.clip(blended, 0, 1)


def get_slices(pred_np, specific_slices, n_slices, vol_depth):
    if specific_slices is not None:
        slices = [max(0, min(s, vol_depth - 1)) for s in specific_slices]
        print(f"Using specific slices: {slices}")
    else:
        fg_per_slice = (pred_np > 0).sum(axis=(0, 1))
        if fg_per_slice.max() > 0:
            slices = sorted(np.argsort(fg_per_slice)[-n_slices:].tolist())
        else:
            slices = np.linspace(
                0, vol_depth-1, n_slices, dtype=int
            ).tolist()
        print(f"Auto-selected slices: {slices}")
    return slices


def visualize_gradcam(img_np, pred_np, gradcam_maps,
                       output_dir, specific_slices=None, n_slices=8):

    slices = get_slices(pred_np, specific_slices,
                         n_slices, img_np.shape[2])

    # Preprocess GradCAM maps
    gc_processed = {}
    for layer_name, gc_map in gradcam_maps.items():
        gc_map = resize_to_match(gc_map, img_np.shape)
        gc_map = normalize_map(gc_map)
        gc_processed[layer_name] = gc_map

    n_layers = len(gc_processed)

    # ── Per-slice plots ────────────────────────────────────────────────────────
    for slice_idx in slices:
        # cols: raw image + pred overlay + one GradCAM per layer
        cols = 1 + 1 + n_layers
        fig, axes = plt.subplots(1, cols,
                                  figsize=(cols * 4, 4.5))
        fig.suptitle(f"GradCAM — Slice z={slice_idx}  "
                     f"[red=high activation | white contour=prediction]",
                     fontsize=11, fontweight="bold")
        fig.patch.set_facecolor("black")

        img_s  = img_np[:, :, slice_idx].T   # (W, H)
        pred_s = pred_np[:, :, slice_idx].T  # (W, H)

        # ── Col 0: raw image ──────────────────────────────────────────────────
        axes[0].imshow(img_s, cmap="gray", origin="upper",
                       vmin=0, vmax=1)
        axes[0].set_title("Image", color="white", fontsize=10)
        axes[0].axis("off")

        # ── Col 1: prediction overlay ─────────────────────────────────────────
        axes[1].imshow(img_s, cmap="gray", origin="upper",
                       vmin=0, vmax=1)
        axes[1].imshow(pred_s, cmap="hot", alpha=0.45,
                       origin="upper", vmin=0, vmax=1)
        # White contour around prediction
        if pred_s.max() > 0:
            contours = measure.find_contours(pred_s, 0.5)
            for c in contours:
                axes[1].plot(c[:, 1], c[:, 0],
                             color=CONTOUR_COLOR,
                             linewidth=CONTOUR_WIDTH,
                             alpha=0.9)
        n_fg = int((pred_s > 0).sum())
        axes[1].set_title(f"Prediction\n(fg={n_fg} vox)",
                          color="white", fontsize=10)
        axes[1].axis("off")

        # ── Cols 2+: GradCAM heatmap per layer ────────────────────────────────
        for k, (layer_name, gc_map) in enumerate(gc_processed.items()):
            gc_s     = gc_map[:, :, slice_idx].T  # (W, H)
            blended  = blend_heatmap(img_s, gc_s, alpha=HEATMAP_ALPHA)

            ax = axes[2 + k]
            ax.imshow(blended, origin="upper")

            # White prediction contour on top
            if pred_s.max() > 0:
                contours = measure.find_contours(pred_s, 0.5)
                for c in contours:
                    ax.plot(c[:, 1], c[:, 0],
                            color=CONTOUR_COLOR,
                            linewidth=CONTOUR_WIDTH,
                            linestyle="--",
                            alpha=0.9)

            parts = layer_name.split(".")
            label = f"GradCAM — Layer {k+1}\n({'.'.join(parts[-2:])})"
            ax.set_title(label, color="white", fontsize=9)
            ax.axis("off")

        plt.tight_layout(pad=0.5)
        out_path = os.path.join(output_dir,
                                f"gradcam_z{slice_idx:03d}.png")
        plt.savefig(out_path, dpi=130, bbox_inches="tight",
                    facecolor="black")
        plt.close()
        print(f"  Saved -> {out_path}")

    # ── MIP summary ────────────────────────────────────────────────────────────
    cols = 1 + 1 + n_layers
    fig, axes = plt.subplots(1, cols, figsize=(cols * 4, 4.5))
    fig.suptitle("GradCAM — Max Intensity Projection",
                 fontsize=11, fontweight="bold")
    fig.patch.set_facecolor("black")

    img_mip  = img_np.max(axis=2).T
    pred_mip = pred_np.max(axis=2).T

    axes[0].imshow(img_mip, cmap="gray", origin="upper")
    axes[0].set_title("Image MIP", color="white"); axes[0].axis("off")

    axes[1].imshow(img_mip, cmap="gray", origin="upper")
    axes[1].imshow(pred_mip, cmap="hot", alpha=0.45, origin="upper")
    if pred_mip.max() > 0:
        contours = measure.find_contours(pred_mip, 0.5)
        for c in contours:
            axes[1].plot(c[:, 1], c[:, 0],
                         color=CONTOUR_COLOR, linewidth=1.0, alpha=0.8)
    axes[1].set_title("Prediction MIP", color="white"); axes[1].axis("off")

    for k, (layer_name, gc_map) in enumerate(gc_processed.items()):
        gc_mip   = gc_map.max(axis=2).T
        blended  = blend_heatmap(img_mip, gc_mip, alpha=HEATMAP_ALPHA)
        ax       = axes[2 + k]
        ax.imshow(blended, origin="upper")
        if pred_mip.max() > 0:
            contours = measure.find_contours(pred_mip, 0.5)
            for c in contours:
                ax.plot(c[:, 1], c[:, 0],
                        color=CONTOUR_COLOR, linewidth=1.0,
                        linestyle="--", alpha=0.8)
        parts = layer_name.split(".")
        ax.set_title(f"GradCAM MIP\n({'.'.join(parts[-2:])})",
                     color="white", fontsize=9)
        ax.axis("off")

    plt.tight_layout(pad=0.5)
    out_path = os.path.join(output_dir, "gradcam_mip.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor="black")
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
    print(f"Z depth: {img.shape[4]} slices — use these indices in SPECIFIC_SLICES")

    # Save middle slice preview
    img_np  = img[0, 0].cpu().numpy()
    mid     = img_np.shape[2] // 2
    plt.figure(figsize=(5, 5), facecolor="black")
    plt.imshow(img_np[:, :, mid].T, cmap="gray", origin="upper")
    plt.title(f"Middle slice z={mid}", color="white")
    plt.axis("off")
    prev_path = os.path.join(OUTPUT_DIR, "preview_middle_slice.png")
    plt.savefig(prev_path, dpi=100, bbox_inches="tight", facecolor="black")
    plt.close()
    print(f"Preview saved -> {prev_path}")

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
    print(f"Prediction: {pred_np.sum():.0f} foreground voxels")

    # ── Compute GradCAM ────────────────────────────────────────────────────────
    gradcam_maps = {}
    for layer_name in TARGET_LAYERS:
        print(f"\nComputing GradCAM: {layer_name}")
        gc_map = run_gradcam_sliding_window(
            model, img, spatial_size,
            target_layer=layer_name,
            device=device, overlap=0.25,
        )
        if gc_map is not None:
            gradcam_maps[layer_name] = gc_map
            print(f"  shape={gc_map.shape} | "
                  f"range=[{gc_map.min():.3f}, {gc_map.max():.3f}]")

    if not gradcam_maps:
        print("\nERROR: No GradCAM maps — available layers:")
        raw = model.module if hasattr(model, "module") else model
        for name, _ in raw.named_modules():
            if "merge" in name or "attention" in name:
                print(f"  {name}")
        return

    # ── Visualize ──────────────────────────────────────────────────────────────
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
