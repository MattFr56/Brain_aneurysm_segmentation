import os
import glob
import shutil
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from os.path import join
from torch.utils import data
from monai.transforms import (
    Compose,
    EnsureType,
    ScaleIntensityRange,
    Resize,
)
from tqdm import tqdm


# ─── UTILS ────────────────────────────────────────────────────────────────────

def is_image_file(filename):
    return any(filename.endswith(ext) for ext in [".nii", ".nii.gz"])


# ─── RAM DISK COPY ────────────────────────────────────────────────────────────

def copy_to_ramdisk(src_dir, ramdisk_dir='/dev/shm/cow_slices'):
    """
    Copy preprocessed .npy files to /dev/shm (RAM-backed filesystem).
    Zero risk to data quality — identical bytes, just faster reads.
    Call once at the start of training before building the dataset.
    Returns the ramdisk directory path to use as the new data root.
    """
    if os.path.exists(ramdisk_dir):
        print(f"RAM disk already populated at {ramdisk_dir} — skipping copy.")
        return ramdisk_dir

    print(f"Copying data to RAM disk ({ramdisk_dir}) ...")
    shutil.copytree(src_dir, ramdisk_dir)
    n = len(glob.glob(os.path.join(ramdisk_dir, '*.npy')))
    print(f"  Done — {n} slices available in RAM disk.")
    return ramdisk_dir


# ─── PREPROCESSING ────────────────────────────────────────────────────────────

def preprocess_to_npy(file_dir, output_dir, patch_size=(512, 512),
                      hu_min=100, hu_max=500,
                      min_vessel_voxels=50):
    """
    Load NIfTI volumes, extract axial slices, apply deterministic
    preprocessing (normalisation + resize) and save as float16 .npy files.

    Optimisations (zero quality impact):
      - Skip background slices with fewer than min_vessel_voxels vessel voxels
        reduces dataset size by ~50-60%, removes useless training signal
      - Save as float16 instead of float32
        halves disk/RAM footprint, precision loss negligible after [0,1] clipping
    """
    os.makedirs(output_dir, exist_ok=True)

    transforms = Compose([
        EnsureType(),
        ScaleIntensityRange(
            a_min=hu_min, a_max=hu_max,
            b_min=0.0,    b_max=1.0,
            clip=True,
        ),
        Resize(spatial_size=patch_size),
    ])

    files = [f for f in os.listdir(file_dir) if is_image_file(f)]
    if len(files) == 0:
        raise RuntimeError(f"No NIfTI files found in '{file_dir}'")

    total_slices = sum(nib.load(join(file_dir, f)).shape[2] for f in files)
    print(f"Preprocessing {len(files)} volumes ({total_slices} slices total)\n"
          f"  source          : {file_dir}\n"
          f"  output          : {output_dir}\n"
          f"  HU window       : [{hu_min}, {hu_max}]\n"
          f"  min vessel voxels: {min_vessel_voxels}\n")

    slice_counter = 0
    skipped_blank = 0
    skipped_bg    = 0

    for fname in tqdm(files, desc="Volumes"):
        vol = nib.load(join(file_dir, fname)).get_fdata().astype(np.float32)

        for s in tqdm(range(vol.shape[2]), desc=f"  {fname}", leave=False):
            slice_img = vol[:, :, s]

            # Skip fully blank slices (zero max -> NaN risk)
            if slice_img.max() == 0:
                skipped_blank += 1
                continue

            # Skip slices with too few vessel-range voxels
            # Only removes slices with almost no vessel signal anyway
            vessel_voxels = int(np.sum((slice_img > hu_min) & (slice_img < hu_max)))
            if vessel_voxels < min_vessel_voxels:
                skipped_bg += 1
                continue

            # Apply transforms -> (1, H, W) tensor
            slice_tensor = transforms(slice_img[np.newaxis, ...])

            out_name = (
                fname.replace('.nii.gz', '').replace('.nii', '')
                + f'_slice{s:03d}.npy'
            )
            # Save as float16 — safe after [0,1] normalisation
            np.save(join(output_dir, out_name),
                    slice_tensor.numpy().astype(np.float16))
            slice_counter += 1

    total_skipped = skipped_blank + skipped_bg
    print(f"\nPreprocessing complete:")
    print(f"  Saved   : {slice_counter} slices")
    print(f"  Skipped : {total_skipped} "
          f"({skipped_blank} blank, {skipped_bg} background/no vessels)")
    print(f"  Estimated RAM usage (float16): "
          f"{slice_counter * patch_size[0] * patch_size[1] * 2 / 1e9:.2f} GB")


# ─── DATASET ──────────────────────────────────────────────────────────────────

class DatasetFromFolder2D(data.Dataset):
    """
    Loads preprocessed 2D slices (.npy) for COVER SSL pretraining.

    Speed optimisation: loads entire dataset into RAM at init time.
    With float16 storage and background filtering ~190 CTA = 1.5-2GB RAM.
    Zero quality impact — data is identical to disk version.

    Args:
        filenames  : list of absolute paths to .npy slice files
        shape      : (H, W) output spatial size fed to the model
        preload    : if True (default), load all slices into RAM at init.
                     Set False only if RAM is insufficient.
    """

    def __init__(self, filenames, shape, preload=True):
        super(DatasetFromFolder2D, self).__init__()
        if len(filenames) == 0:
            raise RuntimeError("DatasetFromFolder2D received an empty file list.")

        self.filenames = filenames
        self.shape     = shape
        self.preload   = preload

        if self.preload:
            print(f"Preloading {len(filenames)} slices into RAM ...")
            self.cache = {}
            for path in tqdm(filenames, desc="Caching", leave=False):
                arr = np.load(path)
                # Normalise to (H, W) — drop channel dim if present
                if arr.ndim == 3:
                    arr = arr[0]
                # Store as float16 to minimise RAM footprint
                self.cache[path] = arr.astype(np.float16)
            ram_gb = sum(v.nbytes for v in self.cache.values()) / 1e9
            print(f"  Done — {ram_gb:.2f} GB used in RAM.")

    def __getitem__(self, index):
        path = self.filenames[index]

        if self.preload:
            # Read from RAM — zero disk I/O per epoch
            arr = self.cache[path]
        else:
            # Fallback: read from disk
            arr = np.load(path)
            if arr.ndim == 3:
                arr = arr[0]

        # Always convert to float32 for GPU computation
        img = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0)  # (1, H, W)

        # Resize to target shape if needed
        if tuple(img.shape[1:]) != tuple(self.shape):
            img = F.interpolate(
                img.unsqueeze(0),           # (1, 1, H, W)
                size=tuple(self.shape),
                mode='bilinear',
                align_corners=False,
            ).squeeze(0)                    # (1, H, W)

        return img                          # float32 tensor (1, H, W)

    def __len__(self):
        return len(self.filenames)
