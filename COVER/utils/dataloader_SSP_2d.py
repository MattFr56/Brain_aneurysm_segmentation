from os.path import join
from os import listdir
import os
import cv2
from torch.utils import data
import numpy as np
import nibabel as nib
from monai.transforms import Compose, ScaleIntensityRanged, RandAdjustContrast, RandGaussianNoise, Resize
from monai.filters import frangi

def is_image_file(filename):
        return any(filename.endswith(extension) for extension in [".nii", ".nii.gz"])

# def preprocess_to_npy(file_dir, output_dir):
#     os.makedirs(output_dir, exist_ok=True)
#     for fname in os.listdir(file_dir):
#         if is_image_file(fname):
#             vol = nib.load(join(file_dir, fname)).get_fdata()  # (512, 512, 128)
#             # Save each slice as a separate .npy file
#             for s in range(vol.shape[2]):
#                 slice_img = vol[:, :, s]
#                 out_name = fname.replace('.nii.gz', '').replace('.nii', '') + f'_slice{s:03d}.npy'
#                 np.save(join(output_dir, out_name), slice_img)

def preprocess_to_npy(file_dir, output_dir, patch_size=(256, 256)):
    os.makedirs(output_dir, exist_ok=True)
    transforms = Compose([
        ScaleIntensityRanged(a_min=100, a_max=500, b_min=0.0, b_max=1.0, clip=True),
        RandAdjustContrast(prob=0.5, gamma=(0.8, 1.2)),
        RandGaussianNoise(prob=0.3, std=0.05),
        Resize(spatial_size=patch_size),
    ])

    for fname in os.listdir(file_dir):
        if is_image_file(fname):
            vol = nib.load(join(file_dir, fname)).get_fdata()
            for s in range(vol.shape[2]):
                slice_img = vol[:, :, s]
                # Prétraitement MONAI
                slice_img = transforms(slice_img[np.newaxis, ...])[0]
        
                # Sauvegarde
                out_name = fname.replace('.nii.gz', '').replace('.nii', '') + f'_slice{s:03d}.npy'
                np.save(join(output_dir, out_name), slice_img)
                
class DatasetFromFolder2D(data.Dataset):
    def __init__(self, filenames, shape):
        super(DatasetFromFolder2D, self).__init__()
        self.filenames = filenames  # list of .npy paths
        self.shape = shape

    def __getitem__(self, index):
        img = np.load(self.filenames[index])
        mask = np.load(self.filenames[index].replace('.npy'))  # Charge le masque
        img = torch.from_numpy(img).unsqueeze(0)  # Ajoute la dimension channel
        mask = torch.from_numpy(mask).unsqueeze(0)
        return img  # Retourne aussi le masque

    # def __getitem__(self, index):
    #     img = np.load(self.filenames[index])     # instant load
    #     img = cv2.resize(img, self.shape)
    #     img = img / img.max()
    #     img = img.astype(np.float32)
    #     img = img[np.newaxis, :, :]
    #     return img

    def __len__(self):
        return len(self.filenames)

# class DatasetFromFolder2D(data.Dataset):
#     def __init__(self, file_dir, filenames, shape):
#         super(DatasetFromFolder2D, self).__init__()
#         self.file_dir = file_dir
#         self.shape = shape
        
#         # Build a flat list of (filename, slice_index) pairs
#         self.slices = []
#         for fname in filenames:
#             img = nib.load(join(file_dir, fname))
#             n_slices = img.shape[2]  # 128
#             for s in range(n_slices):
#                 self.slices.append((fname, s))

#     def __getitem__(self, index):
#         fname, slice_idx = self.slices[index]
#         img = nib.load(join(self.file_dir, fname))
#         img = img.get_fdata()[:, :, slice_idx]   # (512, 512)
#         img = cv2.resize(img, self.shape)
#         img = img / img.max()
#         img = img.astype(np.float32)
#         img = img[np.newaxis, :, :]               # (1, H, W)
#         return img

#     def __len__(self):
#         return len(self.slices)

# class DatasetFromFolder2D(data.Dataset):
#     def __init__(self, file_dir, shape):
#         super(DatasetFromFolder2D, self).__init__()
#         self.filenames = [x for x in listdir(file_dir) if is_image_file(x)]
#         self.file_dir = file_dir
#         self.shape = shape

#     def __getitem__(self, index):
#         img = nib.load(join(self.file_dir, self.filenames[index]))
#         img = img.get_fdata()                    # load as numpy array
#         img = cv2.resize(img, self.shape)
#         img = img / img.max()                    # normalize, 255 doesn't apply to NIfTI
#         img = img.astype(np.float32)
#         img = img[np.newaxis, :, :]

#         return img

#     def __len__(self):
#         return len(self.filenames)