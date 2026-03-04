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

def preprocess_to_npy(file_dirs, output_dir, patch_size=(512, 512),
                      hu_min=100, hu_max=500,
                      min_vessel_voxels=50):
    """
    Load NIfTI volumes from one or multiple folders, extract axial slices,
    apply deterministic preprocessing and save as float16 .npy files.

    Args:
        file_dirs : str or list of str — one or multiple folders containing .nii/.nii.gz
                    e.g. file_dirs='/data/folder1'
                    or   file_dirs=['/data/folder1', '/data/folder2']
                    Files with identical names across folders are disambiguated
                    automatically using their parent folder name as prefix.

    Optimisations (zero quality impact):
      - Skip background slices with fewer than min_vessel_voxels vessel voxels
      - Save as float16 — halves RAM footprint, safe after [0,1] clipping
    """
    os.makedirs(output_dir, exist_ok=True)

    # Normalise to list — works for both single folder and multiple folders
    if isinstance(file_dirs, str):
        file_dirs = [file_dirs]

    # Build flat list of (full_path, unique_stem) pairs
    # unique_stem = foldername__filename to avoid name collisions across folders
    all_files = []
    for folder in file_dirs:
        folder_name = os.path.basename(os.path.normpath(folder))
        for fname in os.listdir(folder):
            if is_image_file(fname):
                full_path   = join(folder, fname)
                unique_stem = f"{folder_name}__{fname.replace('.nii.gz', '').replace('.nii', '')}"
                all_files.append((full_path, unique_stem))

    if len(all_files) == 0:
        raise RuntimeError(f"No NIfTI files found in: {file_dirs}")

    # Check for duplicate stems after prefixing (should never happen but safety check)
    stems = [s for _, s in all_files]
    if len(stems) != len(set(stems)):
        raise RuntimeError("Duplicate file stems detected after prefixing — check folder names.")

    total_slices = sum(nib.load(p).shape[2] for p, _ in all_files)
    print(f"Preprocessing {len(all_files)} volumes ({total_slices} slices total)")
    for folder in file_dirs:
        n = sum(1 for _, s in all_files if s.startswith(os.path.basename(os.path.normpath(folder))))
        print(f"  {folder}  ({n} volumes)")
    print(f"  output           : {output_dir}")
    print(f"  HU window        : [{hu_min}, {hu_max}]")
    print(f"  min vessel voxels: {min_vessel_voxels}\n")

    slice_counter = 0
    skipped_blank = 0
    skipped_bg    = 0

    transforms = Compose([
        EnsureType(),
        ScaleIntensityRange(
            a_min=hu_min, a_max=hu_max,
            b_min=0.0,    b_max=1.0,
            clip=True,
        ),
        Resize(spatial_size=patch_size),
    ])

    for full_path, unique_stem in tqdm(all_files, desc="Volumes"):
        vol = nib.load(full_path).get_fdata().astype(np.float32)

        for s in tqdm(range(vol.shape[2]), desc=f"  {unique_stem}", leave=False):
            slice_img = vol[:, :, s]

            # Skip fully blank slices
            if slice_img.max() == 0:
                skipped_blank += 1
                continue

            # Skip near-background slices with almost no vessel signal
            vessel_voxels = int(np.sum((slice_img > hu_min) & (slice_img < hu_max)))
            if vessel_voxels < min_vessel_voxels:
                skipped_bg += 1
                continue

            slice_tensor = transforms(slice_img[np.newaxis, ...])  # (1, H, W)

            out_name = unique_stem + f'_slice{s:03d}.npy'
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
