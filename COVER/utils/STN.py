import torch
import torch.nn as nn
import torch.nn.functional as nnf
import numpy as np


class SpatialTransformer(nn.Module):
    def __init__(self):
        super(SpatialTransformer, self).__init__()

    def forward(self, src, flow, mode='bilinear'):
        shape = flow.shape[2:]
        vectors = [torch.arange(0, s) for s in shape]

        # FIX 1: explicit indexing='ij' — deprecation fix (same as train script)
        grids = torch.meshgrid(vectors, indexing='ij')
        grid  = torch.stack(grids)          # (2, H, W) or (3, D, H, W)
        grid  = torch.unsqueeze(grid, 0)    # (1, 2, H, W)
        grid  = grid.type(flow.dtype).to(src.device)

        new_locs = grid + flow
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        # FIX 2: align_corners=True — must match the coordinate normalisation above
        # The formula  2*(x/(s-1) - 0.5)  is exactly the align_corners=True convention
        return nnf.grid_sample(src, new_locs, mode=mode, align_corners=True)


class AffineTransformer(nn.Module):
    """3-D Affine Transformer"""

    def __init__(self):
        super().__init__()

    def forward(self, src, mat, mode='bilinear'):
        norm = torch.tensor(
            [[1, 1, 1, src.shape[2]],
             [1, 1, 1, src.shape[3]],
             [1, 1, 1, src.shape[4]]],
            dtype=torch.float
        ).cuda()
        norm     = norm[np.newaxis, :, :]
        mat_new  = mat / norm

        # FIX 3: align_corners=True in both affine_grid and grid_sample — must match
        grid = nnf.affine_grid(
            mat_new,
            [src.shape[0], 3, src.shape[2], src.shape[3], src.shape[4]],
            align_corners=True   # ✅
        )
        return nnf.grid_sample(src, grid, mode=mode, align_corners=True)  # ✅


class AffineTransformer2D(nn.Module):
    """2-D Affine Transformer"""

    def __init__(self):
        super().__init__()

    def forward(self, src, mat, mode='bilinear'):
        # FIX 4: align_corners=True in both affine_grid and grid_sample — must match
        grid = nnf.affine_grid(
            mat,
            [src.shape[0], 2, src.shape[2], src.shape[3]],
            align_corners=True   # ✅
        )
        return nnf.grid_sample(src, grid, mode=mode, align_corners=True)  # ✅        return nnf.grid_sample(src, grid, mode=mode)
