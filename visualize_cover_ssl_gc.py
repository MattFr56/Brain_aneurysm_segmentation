"""
COVER SSL Visualization
=======================
Visualizes what the COVER SSL model learned:
1. Flow fields — deformation between slice pairs
2. Warped images — how well the model aligns slices
3. Feature maps — what the encoder activates on
4. Loss curves — training progress

Usage in Kaggle/Colab:
    Run each cell independently
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import os
import sys

# ─── CELL 0 — Mount Drive ─────────────────────────────────────────────────────
# NOTE: Drive must be mounted BEFORE running this script.
# In your Colab notebook, run this in a separate cell first:
#   from google.colab import drive
#   drive.mount('/content/drive')

import os
if not os.path.exists('/content/drive/MyDrive'):
    raise RuntimeError(
        "Google Drive is not mounted!\n"
        "Run this in a Colab cell first:\n"
        "  from google.colab import drive\n"
        "  drive.mount('/content/drive')"
    )

# Clone repo if not already present
if not os.path.exists('/content/Brain_aneurysm_segmentation'):
    os.system('git clone https://github.com/MattFr56/Brain_aneurysm_segmentation.git '
              '/content/Brain_aneurysm_segmentation')

# ─── CELL 1 — Setup ───────────────────────────────────────────────────────────
# Adjust paths as needed

SSL_CHECKPOINT = '/content/drive/MyDrive/model_final_best.pth.tar'
NPZ_PATH       = '/content/drive/MyDrive/CoW_preprocessed.npz'
COVER_PATH     = '/content/Brain_aneurysm_segmentation/COVER'
CSV_PATH       = '/content/drive/MyDrive/COVER_2D_CoW.csv'
OUTPUT_DIR     = '/content/drive/MyDrive/CoW_visualizations'
os.makedirs(OUTPUT_DIR, exist_ok=True)

sys.path.insert(0, COVER_PATH)
from models.cover import COVER
from utils.STN import SpatialTransformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ─── CELL 2 — Load model ──────────────────────────────────────────────────────

def load_model(ckpt_path):
    model = COVER(n_channels=1, dimensions='2D')
    ckpt  = torch.load(ckpt_path, map_location='cpu')
    state = ckpt.get('state_dict', ckpt)
    # Remove 'module.' prefix if saved with DataParallel
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model.to(device)

model = load_model(SSL_CHECKPOINT)
stn   = SpatialTransformer().to(device)
print(f"✅ Model loaded from {SSL_CHECKPOINT}")


# ─── CELL 3 — Load sample slices ──────────────────────────────────────────────

def load_slice(npz, key):
    arr = npz[key].astype(np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W)

npz  = np.load(NPZ_PATH)
keys = sorted(npz.files)
print(f"✅ Loaded {len(keys)} slices from npz")

# Pick a moving and fixed slice from the same patient
# Try to find two slices from same volume (adjacent slices)
patient_keys = {}
for k in keys:
    stem = '_'.join(k.split('_')[:-1])
    patient_keys.setdefault(stem, []).append(k)

# Pick patient with most slices (most likely to be CoW region)
best_patient = max(patient_keys, key=lambda p: len(patient_keys[p]))
patient_slices = sorted(patient_keys[best_patient])
print(f"Using patient: {best_patient} ({len(patient_slices)} slices)")

# Pick middle slices (most likely to contain CoW)
mid   = len(patient_slices) // 2
key_M = patient_slices[mid]
key_F = patient_slices[mid + 5] if mid + 5 < len(patient_slices) else patient_slices[mid - 5]

img_M = load_slice(npz, key_M)
img_F = load_slice(npz, key_F)
print(f"Moving: {key_M}")
print(f"Fixed : {key_F}")


# ─── CELL 4 — Run inference ───────────────────────────────────────────────────

with torch.no_grad():
    flow_pred, feat_M, feat_F = model(M=img_M, F=img_F)
    warped = stn(img_M, flow_pred)

flow_np   = flow_pred[0].cpu().numpy()   # (2, H, W)
img_M_np  = img_M[0, 0].cpu().numpy()   # (H, W)
img_F_np  = img_F[0, 0].cpu().numpy()   # (H, W)
warped_np = warped[0, 0].cpu().numpy()  # (H, W)
feat_M_np = feat_M[0].cpu().numpy()     # (C, H, W)
feat_F_np = feat_F[0].cpu().numpy()     # (C, H, W)

print(f"✅ Inference done")
print(f"   Flow range: [{flow_np.min():.2f}, {flow_np.max():.2f}] pixels")
print(f"   Feature channels: {feat_M_np.shape[0]}")


# ─── CELL 5 — Plot 1: Warped image comparison ─────────────────────────────────

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle('COVER SSL — Image Registration', fontsize=14, fontweight='bold')

axes[0].imshow(img_M_np,  cmap='gray', vmin=0, vmax=1)
axes[0].set_title('Moving (M)', fontsize=12)
axes[0].axis('off')

axes[1].imshow(img_F_np,  cmap='gray', vmin=0, vmax=1)
axes[1].set_title('Fixed (F)', fontsize=12)
axes[1].axis('off')

axes[2].imshow(warped_np, cmap='gray', vmin=0, vmax=1)
axes[2].set_title('Warped M→F', fontsize=12)
axes[2].axis('off')

diff = np.abs(img_F_np - warped_np)
im = axes[3].imshow(diff, cmap='hot', vmin=0, vmax=0.3)
axes[3].set_title(f'|F - Warped| (mean={diff.mean():.4f})', fontsize=12)
axes[3].axis('off')
plt.colorbar(im, ax=axes[3], fraction=0.046)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '1_warped_comparison.png'), dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved 1_warped_comparison.png")


# ─── CELL 6 — Plot 2: Flow field visualization ────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('COVER SSL — Predicted Flow Field', fontsize=14, fontweight='bold')

# Flow magnitude
flow_mag = np.sqrt(flow_np[0]**2 + flow_np[1]**2)
im0 = axes[0].imshow(flow_mag, cmap='viridis')
axes[0].set_title('Flow Magnitude (pixels)', fontsize=12)
axes[0].axis('off')
plt.colorbar(im0, ax=axes[0], fraction=0.046)

# Flow x component
im1 = axes[1].imshow(flow_np[0], cmap='RdBu', norm=Normalize(-flow_mag.max(), flow_mag.max()))
axes[1].set_title('Flow X (horizontal)', fontsize=12)
axes[1].axis('off')
plt.colorbar(im1, ax=axes[1], fraction=0.046)

# Flow y component
im2 = axes[2].imshow(flow_np[1], cmap='RdBu', norm=Normalize(-flow_mag.max(), flow_mag.max()))
axes[2].set_title('Flow Y (vertical)', fontsize=12)
axes[2].axis('off')
plt.colorbar(im2, ax=axes[2], fraction=0.046)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '2_flow_field.png'), dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved 2_flow_field.png")


# ─── CELL 7 — Plot 3: Flow arrows (quiver) ────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('COVER SSL — Flow Arrows', fontsize=14, fontweight='bold')

step = 16  # arrow density
H, W = flow_np.shape[1:]
Y, X = np.mgrid[0:H:step, 0:W:step]
U    = flow_np[1, ::step, ::step]  # x displacement
V    = -flow_np[0, ::step, ::step]  # y displacement (flip for display)

for ax, img, title in [(axes[0], img_M_np, 'Flow on Moving'),
                        (axes[1], img_F_np, 'Flow on Fixed')]:
    ax.imshow(img, cmap='gray', vmin=0, vmax=1, alpha=0.8)
    ax.quiver(X, Y, U, V,
              color='red', scale=50, scale_units='inches',
              width=0.003, alpha=0.8)
    ax.set_title(title, fontsize=12)
    ax.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '3_flow_arrows.png'), dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved 3_flow_arrows.png")


# ─── CELL 8 — Plot 4: Feature maps ───────────────────────────────────────────

fig = plt.figure(figsize=(20, 8))
fig.suptitle('COVER SSL — Encoder Feature Maps', fontsize=14, fontweight='bold')

n_show = min(8, feat_M_np.shape[0])
gs     = gridspec.GridSpec(2, n_show + 1, figure=fig)

# Original image
ax_orig = fig.add_subplot(gs[:, 0])
ax_orig.imshow(img_M_np, cmap='gray', vmin=0, vmax=1)
ax_orig.set_title('Input', fontsize=10)
ax_orig.axis('off')

# Feature channels
for i in range(n_show):
    feat = feat_M_np[i]
    feat_norm = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)

    ax = fig.add_subplot(gs[0, i + 1])
    ax.imshow(feat_norm, cmap='hot')
    ax.set_title(f'Ch {i}', fontsize=8)
    ax.axis('off')

    # Overlay on image
    ax2 = fig.add_subplot(gs[1, i + 1])
    ax2.imshow(img_M_np, cmap='gray', vmin=0, vmax=1, alpha=0.6)
    ax2.imshow(feat_norm, cmap='hot', alpha=0.5)
    ax2.set_title(f'Overlay', fontsize=8)
    ax2.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '4_feature_maps.png'), dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved 4_feature_maps.png")


# ─── CELL 9 — Plot 5: Mean feature activation ────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('COVER SSL — Mean Feature Activation', fontsize=14, fontweight='bold')

# Mean activation across all channels
feat_M_mean = feat_M_np.mean(axis=0)
feat_F_mean = feat_F_np.mean(axis=0)
feat_M_norm = (feat_M_mean - feat_M_mean.min()) / (feat_M_mean.max() - feat_M_mean.min() + 1e-8)
feat_F_norm = (feat_F_mean - feat_F_mean.min()) / (feat_F_mean.max() - feat_F_mean.min() + 1e-8)

axes[0].imshow(img_M_np, cmap='gray', vmin=0, vmax=1)
axes[0].set_title('Input slice', fontsize=12)
axes[0].axis('off')

axes[1].imshow(img_M_np, cmap='gray', vmin=0, vmax=1, alpha=0.5)
im = axes[1].imshow(feat_M_norm, cmap='hot', alpha=0.6)
axes[1].set_title('Encoder activation (Moving)', fontsize=12)
axes[1].axis('off')
plt.colorbar(im, ax=axes[1], fraction=0.046)

axes[2].imshow(img_F_np, cmap='gray', vmin=0, vmax=1, alpha=0.5)
im2 = axes[2].imshow(feat_F_norm, cmap='hot', alpha=0.6)
axes[2].set_title('Encoder activation (Fixed)', fontsize=12)
axes[2].axis('off')
plt.colorbar(im2, ax=axes[2], fraction=0.046)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '5_mean_activation.png'), dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved 5_mean_activation.png")


# ─── CELL 10 — Plot 6: Loss curves ───────────────────────────────────────────

if os.path.isfile(CSV_PATH):
    import pandas as pd
    df = pd.read_csv(CSV_PATH)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('COVER SSL — Training Curves', fontsize=14, fontweight='bold')

    # Flow loss
    axes[0].plot(df['Train Loss_vec'], label='Train', color='blue',  linewidth=2)
    axes[0].plot(df['Val Loss_vec'],   label='Val',   color='orange', linewidth=2)
    axes[0].set_title('Flow Loss (MAE)', fontsize=12)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Contrastive loss
    axes[1].plot(df['Train Loss_con'], label='Train', color='blue',  linewidth=2)
    axes[1].plot(df['Val Loss_con'],   label='Val',   color='orange', linewidth=2)
    axes[1].set_title('Contrastive Loss (Cosine)', fontsize=12)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '6_loss_curves.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Saved 6_loss_curves.png")
else:
    print(f"⚠️  CSV not found at {CSV_PATH} — skipping loss curves")

print(f"\n✅ All visualizations saved to {OUTPUT_DIR}")
