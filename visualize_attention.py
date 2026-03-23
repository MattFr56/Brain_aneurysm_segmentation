import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from monai.networks.nets import AttentionUnet
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    Orientationd, Spacingd, ScaleIntensityRanged,
    ToTensord,
)
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch

# ── Config ─────────────────────────────────────────────────────────────────────
IMAGE_PATH   = "/content/data/your_volume.nii.gz"   # ← your CTA volume
CHECKPOINT   = "/content/best_model_phase1b.pth"     # ← your checkpoint
OUTPUT_DIR   = "/content/attention_maps"
os.makedirs(OUTPUT_DIR, exist_ok=True)

HU_MIN     = 100
HU_MAX     = 400
SPATIAL_SIZE = (128, 128, 32)
SW_OVERLAP   = 0.5
SW_BATCH     = 2


# ── Hook storage ───────────────────────────────────────────────────────────────
attention_maps = {}
hooks          = []


def make_hook(name):
    """Create a forward hook that saves the attention gate output."""
    def hook(module, input, output):
        # output is the attention-weighted feature map
        # store mean across channels → (H, W, D)
        attention_maps[name] = output.detach().cpu()
    return hook


def register_attention_hooks(model):
    """Register hooks on all attention gate psi layers."""
    raw = model.module if hasattr(model, "module") else model
    for name, module in raw.named_modules():
        # AttentionUnet attention gates have a 'psi' sigmoid output
        if "psi" in name and hasattr(module, "weight"):
            h = module.register_forward_hook(make_hook(name))
            hooks.append(h)
            print(f"  Registered hook: {name}")


def remove_hooks():
    for h in hooks:
        h.remove()
    hooks.clear()


def load_model(checkpoint_path, device):
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
    print(f"✓ Loaded {os.path.basename(checkpoint_path)}")
    return model, threshold, spatial, hu_min, hu_max


def visualize_attention(img_tensor, pred_tensor, attention_maps,
                        output_dir, n_slices=6):
    """
    Plot attention maps alongside image and prediction.
    img_tensor:  (1, H, W, D)
    pred_tensor: (1, H, W, D)
    """
    img  = img_tensor.squeeze().numpy()   # (H, W, D)
    pred = pred_tensor.squeeze().numpy()  # (H, W, D)

    # Pick slices with most foreground
    fg_per_slice = (pred > 0).sum(axis=(0, 1))
    if fg_per_slice.max() > 0:
        top_slices = np.argsort(fg_per_slice)[-n_slices:][::-1]
    else:
        top_slices = np.linspace(0, img.shape[2]-1, n_slices, dtype=int)

    att_names = list(attention_maps.keys())
    n_att     = len(att_names)

    print(f"\nAttention maps captured: {att_names}")
    print(f"Visualizing {n_slices} slices with most foreground")

    for slice_idx in top_slices:
        fig, axes = plt.subplots(
            1, 2 + n_att,
            figsize=((2 + n_att) * 4, 4)
        )
        fig.suptitle(f"Slice z={slice_idx} — Image | Pred | Attention Gates",
                     fontsize=12)

        # Image
        axes[0].imshow(img[:, :, slice_idx].T,
                       cmap="gray", origin="lower",
                       vmin=0, vmax=1)
        axes[0].set_title("Image")
        axes[0].axis("off")

        # Prediction overlay
        axes[1].imshow(img[:, :, slice_idx].T,
                       cmap="gray", origin="lower",
                       vmin=0, vmax=1)
        axes[1].imshow(pred[:, :, slice_idx].T,
                       cmap="hot", alpha=0.5, origin="lower")
        axes[1].set_title("Prediction")
        axes[1].axis("off")

        # Attention maps per gate
        for k, att_name in enumerate(att_names):
            att = attention_maps[att_name]  # (B, C, H, W, D) or similar

            # Average over batch and channel dims
            if att.dim() == 5:
                att_map = att[0].mean(0)  # (H, W, D)
            elif att.dim() == 4:
                att_map = att[0]          # (H, W, D)
            else:
                att_map = att.mean(0)

            att_np = att_map.numpy()

            # Resize to match image if needed
            if att_np.shape != img.shape:
                from scipy.ndimage import zoom
                scale  = [img.shape[i] / att_np.shape[i]
                           for i in range(3)]
                att_np = zoom(att_np, scale, order=1)

            # Normalize to [0, 1]
            att_min, att_max = att_np.min(), att_np.max()
            if att_max > att_min:
                att_np = (att_np - att_min) / (att_max - att_min)

            ax = axes[2 + k]
            ax.imshow(img[:, :, slice_idx].T,
                      cmap="gray", origin="lower",
                      vmin=0, vmax=1)
            ax.imshow(att_np[:, :, slice_idx].T,
                      cmap="jet", alpha=0.6, origin="lower",
                      vmin=0, vmax=1)
            ax.set_title(f"Attn: {att_name.split('.')[-3]}")
            ax.axis("off")

        plt.tight_layout()
        out_path = os.path.join(output_dir,
                                f"attention_z{slice_idx:03d}.png")
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  Saved -> {out_path}")

    # ── Summary plot — max intensity projection ────────────────────────────────
    fig, axes = plt.subplots(1, 2 + n_att, figsize=((2 + n_att) * 4, 4))
    fig.suptitle("Max Intensity Projection (all slices)", fontsize=12)

    # MIP of image
    axes[0].imshow(img.max(axis=2).T, cmap="gray", origin="lower")
    axes[0].set_title("Image MIP")
    axes[0].axis("off")

    # MIP of prediction
    axes[1].imshow(img.max(axis=2).T, cmap="gray", origin="lower")
    axes[1].imshow(pred.max(axis=2).T, cmap="hot", alpha=0.5,
                   origin="lower")
    axes[1].set_title("Prediction MIP")
    axes[1].axis("off")

    # MIP of each attention map
    for k, att_name in enumerate(att_names):
        att = attention_maps[att_name]
        if att.dim() == 5:
            att_map = att[0].mean(0).numpy()
        else:
            att_map = att[0].numpy() if att.dim() == 4 \
                      else att.mean(0).numpy()

        if att_map.shape != img.shape:
            from scipy.ndimage import zoom
            scale   = [img.shape[i] / att_map.shape[i] for i in range(3)]
            att_map = zoom(att_map, scale, order=1)

        att_min, att_max = att_map.min(), att_map.max()
        if att_max > att_min:
            att_map = (att_map - att_min) / (att_max - att_min)

        ax = axes[2 + k]
        ax.imshow(img.max(axis=2).T, cmap="gray", origin="lower")
        ax.imshow(att_map.max(axis=2).T, cmap="jet", alpha=0.6,
                  origin="lower", vmin=0, vmax=1)
        ax.set_title(f"Attn MIP: {att_name.split('.')[-3]}")
        ax.axis("off")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "attention_mip.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n✓ MIP saved -> {out_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load model ─────────────────────────────────────────────────────────────
    model, threshold, spatial_size, hu_min, hu_max = load_model(
        CHECKPOINT, device
    )

    # ── Register attention hooks ───────────────────────────────────────────────
    register_attention_hooks(model)

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
    img  = data["img"].unsqueeze(0).to(device)  # (1, 1, H, W, D)
    print(f"\nImage shape after preprocessing: {img.shape}")

    # ── Run inference — hooks capture attention maps ───────────────────────────
    model.eval()
    with torch.no_grad():
        output = sliding_window_inference(
            img, roi_size=spatial_size,
            sw_batch_size=SW_BATCH,
            predictor=model, overlap=SW_OVERLAP,
        )

    pred = (torch.sigmoid(output) > threshold).float()
    print(f"Prediction: {pred.sum().int()} foreground voxels")
    print(f"Attention maps captured: {list(attention_maps.keys())}")

    # ── Visualize ──────────────────────────────────────────────────────────────
    visualize_attention(
        img_tensor=img[0].cpu(),
        pred_tensor=pred[0].cpu(),
        attention_maps=attention_maps,
        output_dir=OUTPUT_DIR,
        n_slices=6,
    )

    # ── Cleanup ────────────────────────────────────────────────────────────────
    remove_hooks()
    print(f"\n✓ Done — attention maps saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
