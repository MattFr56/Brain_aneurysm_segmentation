import os
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from os.path import join
from torch.utils import data
from monai.transforms import (
    Compose,
    EnsureType,          # FIX 2: converts numpy → tensor before augmentation
    ScaleIntensityRange, # FIX 1: non-dict version, consistent with array input
    Resize,
)
from tqdm import tqdm


# ─── UTILS ────────────────────────────────────────────────────────────────────

def is_image_file(filename):
    return any(filename.endswith(ext) for ext in [".nii", ".nii.gz"])


# ─── PREPROCESSING ────────────────────────────────────────────────────────────

def preprocess_to_npy(file_dir, output_dir, patch_size=(512, 512)):
    """
    Load NIfTI volumes, extract axial slices, apply deterministic
    preprocessing (normalisation + resize) and save as .npy files.

    FIX 3: Random augmentations (contrast, noise) are intentionally removed
           from here — they must be applied at training time so the model
           sees different augmented versions of each slice every epoch.
    FIX 2: EnsureType() added so MONAI transforms receive a proper tensor.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Deterministic transforms only — no randomness at preprocessing stage
    transforms = Compose([
        EnsureType(),                                                # FIX 2
        ScaleIntensityRange(
            a_min=100, a_max=500,
            b_min=0.0, b_max=1.0,
            clip=True
        ),
        Resize(spatial_size=patch_size),
    ])

    files = [f for f in os.listdir(file_dir) if is_image_file(f)]
    if len(files) == 0:
        raise RuntimeError(f"No NIfTI files found in '{file_dir}'")

    total_slices = sum(nib.load(join(file_dir, f)).shape[2] for f in files)
    print(f"Preprocessing {len(files)} volumes ({total_slices} slices total)\n"
          f"  source : {file_dir}\n"
          f"  output : {output_dir}\n")

    slice_counter = 0
    skipped       = 0

    for fname in tqdm(files, desc="Volumes"):
        vol = nib.load(join(file_dir, fname)).get_fdata().astype(np.float32)

        for s in tqdm(range(vol.shape[2]), desc=f"  {fname}", leave=False):
            slice_img = vol[:, :, s]                     # (H, W)

            # FIX 6: skip blank slices to avoid NaN from division by zero
            if slice_img.max() == 0:
                skipped += 1
                continue

            # Add channel dim expected by MONAI transforms → (1, H, W)
            slice_tensor = transforms(slice_img[np.newaxis, ...])  # (1, H, W)

            out_name = (
                fname.replace('.nii.gz', '').replace('.nii', '')
                + f'_slice{s:03d}.npy'
            )
            # Save as float32 numpy array — squeeze channel dim for compact storage
            np.save(join(output_dir, out_name), slice_tensor.numpy().astype(np.float32))
            slice_counter += 1

    print(f"\nPreprocessing complete: {slice_counter} slices saved"
          f" ({skipped} blank slices skipped)")


# ─── DATASET ──────────────────────────────────────────────────────────────────

class DatasetFromFolder2D(data.Dataset):
    """
    Loads preprocessed 2D slices (.npy) for COVER SSL pretraining.

    Each .npy file contains a single (H, W) float32 array (channel dim was
    squeezed at save time). __getitem__ restores the channel dim and resizes
    to self.shape so the batch shape is always (1, H, W).

    FIX 4: explicit cast to float32 — avoids float64 tensor instability.
    FIX 5: self.shape is now actually used — bilinear resize in __getitem__.
    """

    def __init__(self, filenames, shape):
        """
        Args:
            filenames : list of absolute paths to .npy slice files
            shape     : (H, W) tuple — output spatial size fed to the model
        """
        super(DatasetFromFolder2D, self).__init__()
        if len(filenames) == 0:
            raise RuntimeError("DatasetFromFolder2D received an empty file list.")
        self.filenames = filenames
        self.shape     = shape  # e.g. (384, 384) = int(256 * 1.5)

    def __getitem__(self, index):
        # Load — shape on disk is either (H, W) or (1, H, W) depending on how
        # preprocess_to_npy saved it. Normalise to (1, H, W) either way.
        img = np.load(self.filenames[index])

        # FIX 4: explicit float32 cast
        img = img.astype(np.float32)

        # Normalise to (1, H, W) regardless of saved shape
        if img.ndim == 2:
            img = torch.from_numpy(img).unsqueeze(0)   # (H, W)   → (1, H, W)
        elif img.ndim == 3:
            img = torch.from_numpy(img)                # (1, H, W) already fine
        else:
            raise ValueError(f"Unexpected array shape {img.shape} in {self.filenames[index]}")

        # FIX 5: resize to self.shape if spatial dims don't match
        if tuple(img.shape[1:]) != tuple(self.shape):
            img = F.interpolate(
                img.unsqueeze(0),          # (1, 1, H, W) — N, C, H, W
                size=tuple(self.shape),
                mode='bilinear',
                align_corners=False,
            ).squeeze(0)                   # back to (1, H, W)

        return img                         # float32 tensor (1, H, W)

    def __len__(self):
        return len(self.filenames)
