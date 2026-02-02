from __future__ import annotations
from typing import Optional

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

def signed_distance_2d(mask: np.ndarray) -> np.ndarray:
    """Signed distance as in Listing 1: sdf = d_out - d_in.
    mask: 2D bool or {0,1}
    Returns float32 sdf with positive outside, negative inside.
    """
    m = mask.astype(bool)
    pos = distance_transform_edt(m)         # d_in (>=0 inside)
    neg = distance_transform_edt(~m)        # d_out (>=0 outside)
    sdf = neg - pos
    return sdf.astype(np.float32)

def signed_distance_nd(mask: np.ndarray) -> np.ndarray:
    """N-D signed distance (works for 2D/3D)."""
    m = mask.astype(bool)
    pos = distance_transform_edt(m)
    neg = distance_transform_edt(~m)
    return (neg - pos).astype(np.float32)

def multiclass_sdf_targets(labels: torch.Tensor, num_classes: int, normalize: bool = True) -> torch.Tensor:
    """Compute per-class signed distance targets.
    labels: (N, H, W) or (N, D, H, W)
    Returns: (N, C, *spatial) float32, optionally normalized to [-1,1].
    """
    device = labels.device
    labs = labels.detach().cpu().numpy()
    outs = []
    for n in range(labs.shape[0]):
        lab = labs[n]
        c_sdfs = []
        for c in range(num_classes):
            m = (lab == c).astype(np.uint8)
            sdf = signed_distance_nd(m)
            if normalize:
                mx = np.max(np.abs(sdf)) + 1e-6
                sdf = sdf / mx  # now roughly in [-1,1]
            c_sdfs.append(sdf)
        outs.append(np.stack(c_sdfs, axis=0))
    return torch.from_numpy(np.stack(outs, axis=0)).float().to(device)
