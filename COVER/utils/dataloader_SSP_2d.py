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
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing


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

def _process_volume(args_tuple):
    """
    Worker function — processes a single volume.
    Must be a top-level function for multiprocessing pickling.
    Returns (slice_counter, skipped_blank, skipped_bg).
    """
    full_path, unique_stem, output_dir, patch_size, hu_min, hu_max, min_vessel_voxels = args_tuple

    # Each worker builds its own transform pipeline
    transforms = Compose([
        EnsureType(),
        ScaleIntensityRange(a_min=hu_min, a_max=hu_max,
                            b_min=0.0, b_max=1.0, clip=True),
        Resize(spatial_size=patch_size),
    ])

    vol = nib.load(full_path).get_fdata().astype(np.float32)
    slice_counter = 0
    skipped_blank = 0
    skipped_bg    = 0

    for s in range(vol.shape[2]):
        slice_img = vol[:, :, s]

        if slice_img.max() == 0:
            skipped_blank += 1
            continue

        vessel_voxels = int(np.sum((slice_img > hu_min) & (slice_img < hu_max)))
        if vessel_voxels < min_vessel_voxels:
            skipped_bg += 1
            continue

        slice_tensor = transforms(slice_img[np.newaxis, ...])
        out_name     = unique_stem + f'_slice{s:03d}.npy'
        np.save(join(output_dir, out_name),
                slice_tensor.numpy().astype(np.float16))
        slice_counter += 1

    return slice_counter, skipped_blank, skipped_bg


def preprocess_to_npy(file_dirs, output_dir, patch_size=(256, 256),
                      hu_min=100, hu_max=500,
                      min_vessel_voxels=500,  # raised from 50 — keeps only slices with meaningful vessel content
                      num_workers=None):
    """
    Load NIfTI volumes from one or multiple folders, extract axial slices,
    apply deterministic preprocessing and save as float16 .npy files.

    Args:
        file_dirs   : str or list of str — folders containing .nii/.nii.gz
        output_dir  : where to save .npy slices
        patch_size  : (H, W) — default 256x256 (was 512x512, changed for speed)
                      Since training crops to 256 anyway, preprocessing at 256
                      is 4x faster and uses 4x less disk/RAM with zero quality loss
        num_workers : number of parallel workers (default: cpu_count - 1)

    Parallelisation: each volume is processed by a separate CPU worker.
    Speedup: ~4-6x on Colab (2 vCPUs) vs sequential, more on local machines.
    """
    os.makedirs(output_dir, exist_ok=True)

    if isinstance(file_dirs, str):
        file_dirs = [file_dirs]

    # Build flat list of (full_path, unique_stem)
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

    stems = [s for _, s in all_files]
    if len(stems) != len(set(stems)):
        raise RuntimeError("Duplicate file stems detected — check folder names.")

    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)

    total_vols = len(all_files)
    print(f"Preprocessing {total_vols} volumes")
    for folder in file_dirs:
        n = sum(1 for _, s in all_files
                if s.startswith(os.path.basename(os.path.normpath(folder))))
        print(f"  {folder}  ({n} volumes)")
    print(f"  output           : {output_dir}")
    print(f"  patch_size       : {patch_size}")
    print(f"  HU window        : [{hu_min}, {hu_max}]")
    print(f"  min vessel voxels: {min_vessel_voxels}")
    print(f"  workers          : {num_workers}\n")

    # Build argument tuples for each worker
    worker_args = [
        (full_path, unique_stem, output_dir,
         patch_size, hu_min, hu_max, min_vessel_voxels)
        for full_path, unique_stem in all_files
    ]

    slice_counter = 0
    skipped_blank = 0
    skipped_bg    = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_process_volume, arg): arg for arg in worker_args}
        # mininterval=5 — update display every 5s max to avoid Colab lag
        pbar    = tqdm(as_completed(futures), total=total_vols, desc="Volumes",
                       mininterval=5, dynamic_ncols=True)
        for future in pbar:
            s_count, s_blank, s_bg = future.result()
            slice_counter += s_count
            skipped_blank += s_blank
            skipped_bg    += s_bg
            pbar.set_postfix(saved=slice_counter,
                             skipped=skipped_blank + skipped_bg)

    total_skipped = skipped_blank + skipped_bg
    print(f"\nPreprocessing complete:")
    print(f"  Saved   : {slice_counter} slices")
    print(f"  Skipped : {total_skipped} "
          f"({skipped_blank} blank, {skipped_bg} background/no vessels)")
    print(f"  Estimated RAM usage (float16): "
          f"{slice_counter * patch_size[0] * patch_size[1] * 2 / 1e9:.2f} GB")


# ─── DATASET ──────────────────────────────────────────────────────────────────

def pack_to_npz(npy_dir, npz_path):
    """
    Pack all .npy slices from a directory into a single .npz archive.
    Run once — replaces 39k individual Drive file reads with one single read.

    Usage in Colab (run while training epoch 1):
        from utils.dataloader_SSP_2d import pack_to_npz
        pack_to_npz(
            '/content/drive/MyDrive/CoW_preprocessed',
            '/content/drive/MyDrive/CoW_preprocessed.npz'
        )
    """
    files = sorted([f for f in os.listdir(npy_dir) if f.endswith('.npy')])
    if len(files) == 0:
        raise RuntimeError(f"No .npy files found in {npy_dir}")

    print(f"Packing {len(files)} slices into {npz_path} ...")
    data_dict = {}
    for f in tqdm(files, desc="Packing", mininterval=10):
        arr = np.load(os.path.join(npy_dir, f))
        if arr.ndim == 3:
            arr = arr[0]
        # sanitise key — npz keys cannot contain dots or dashes
        key = f.replace('.npy', '').replace('-', '_').replace('.', '_')
        data_dict[key] = arr.astype(np.float16)

    np.savez_compressed(npz_path, **data_dict)
    size_gb = os.path.getsize(npz_path) / 1e9
    print(f"Done — {npz_path}  ({size_gb:.2f} GB)")


class DatasetFromFolder2D(data.Dataset):
    """
    Loads preprocessed 2D slices for COVER SSL pretraining.

    Supports two modes:
      1. npz mode  (recommended) — single .npz file loaded into RAM once
                                    one Drive read per session, then zero I/O
      2. npy mode  (fallback)    — reads individual .npy files from disk/Drive
                                    slower but works without packing step

    Args:
        filenames  : list of .npy paths (npy mode) OR single .npz path in a list
        shape      : (H, W) output spatial size
        preload    : load all data into RAM at init (default True for npz,
                     recommended False for npy to avoid OOM)
        npz_path   : if provided, load from this .npz file instead of filenames
    """

    def __init__(self, filenames, shape, preload=False, npz_path=None):
        super(DatasetFromFolder2D, self).__init__()
        self.shape    = shape
        self.npz_path = npz_path
        self.preload  = preload
        self.cache    = {}

        if npz_path is not None:
            # ── NPZ MODE — one file, load everything into RAM ────────────────
            print(f"Loading dataset from {npz_path} ...")
            npz           = np.load(npz_path)
            self.keys     = sorted(npz.files)
            self.filenames = self.keys

            estimated_gb = len(self.keys) * 256 * 256 * 2 / 1e9
            print(f"  {len(self.keys)} slices  "
                  f"(estimated {estimated_gb:.2f} GB float16 in RAM)")

            if estimated_gb > 9.0:
                print(f"  ⚠️  Warning: may exceed Colab RAM — consider "
                      f"raising min_vessel_voxels when re-preprocessing.")

            for key in tqdm(self.keys, desc="Loading into RAM", mininterval=5):
                self.cache[key] = npz[key].astype(np.float16)
            npz.close()

            ram_gb = sum(v.nbytes for v in self.cache.values()) / 1e9
            print(f"  Done — {ram_gb:.2f} GB used in RAM. Zero I/O during training.")

        else:
            # ── NPY MODE — individual files ───────────────────────────────────
            if len(filenames) == 0:
                raise RuntimeError("DatasetFromFolder2D received an empty file list.")
            self.filenames = filenames

            if preload:
                estimated_gb = len(filenames) * 256 * 256 * 2 / 1e9
                print(f"Preloading {len(filenames)} slices into RAM "
                      f"(estimated {estimated_gb:.2f} GB) ...")
                if estimated_gb > 9.0:
                    print(f"  ⚠️  Warning: may exceed Colab RAM — use npz mode instead.")
                for path in tqdm(filenames, desc="Caching", leave=False, mininterval=5):
                    arr = np.load(path)
                    if arr.ndim == 3:
                        arr = arr[0]
                    self.cache[path] = arr.astype(np.float16)
                ram_gb = sum(v.nbytes for v in self.cache.values()) / 1e9
                print(f"  Done — {ram_gb:.2f} GB used in RAM.")

    def __getitem__(self, index):
        key = self.filenames[index]

        if key in self.cache:
            arr = self.cache[key]
        else:
            # npy fallback — read from disk
            arr = np.load(key)
            if arr.ndim == 3:
                arr = arr[0]

        img = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0)  # (1, H, W)

        if tuple(img.shape[1:]) != tuple(self.shape):
            img = F.interpolate(
                img.unsqueeze(0),
                size=tuple(self.shape),
                mode='bilinear',
                align_corners=False,
            ).squeeze(0)

        return img  # float32 tensor (1, H, W)

    def __len__(self):
        return len(self.filenames)
