from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import cv2
import torch

KernelName = Literal["mean","max","min","mode","gaussian","bilateral"]
MorphName = Literal[
    "erosion","dilation","opening","closing","internal_boundary","external_boundary","gradient"
]

def _to_uint8(mask: np.ndarray) -> np.ndarray:
    return (mask.astype(np.uint8) * 255)

def apply_morphology(binary: np.ndarray, op: MorphName) -> np.ndarray:
    """Apply a 3x3 morphological op to a 2D binary mask.
    Note: For 3D volumes, apply slice-wise (D times) at call-site.
    """
    kernel = np.ones((3,3), np.uint8)
    b = _to_uint8(binary)
    if op == "erosion":
        out = cv2.erode(b, kernel, iterations=1)
    elif op == "dilation":
        out = cv2.dilate(b, kernel, iterations=1)
    elif op == "opening":
        out = cv2.morphologyEx(b, cv2.MORPH_OPEN, kernel)
    elif op == "closing":
        out = cv2.morphologyEx(b, cv2.MORPH_CLOSE, kernel)
    elif op == "internal_boundary":
        er = cv2.erode(b, kernel, iterations=1)
        out = cv2.subtract(b, er)
    elif op == "external_boundary":
        di = cv2.dilate(b, kernel, iterations=1)
        out = cv2.subtract(di, b)
    elif op == "gradient":
        out = cv2.morphologyEx(b, cv2.MORPH_GRADIENT, kernel)
    else:
        raise ValueError(f"Unknown morphology op: {op}")
    return (out > 0).astype(np.uint8)

def _local_mode(patch: np.ndarray) -> float:
    # patch is 0/1
    s = patch.sum()
    return float(1.0 if s >= (patch.size/2) else 0.0)

def aggregate_kernel(mask: np.ndarray, kernel: KernelName, image: Optional[np.ndarray] = None) -> np.ndarray:
    """Aggregate local neighborhood frequencies into an adaptive map.
    mask: 2D uint8 {0,1}
    image: 2D float32, required for bilateral
    """
    k = 3
    if kernel == "mean":
        return cv2.blur(mask.astype(np.float32), (k,k))
    if kernel == "max":
        return cv2.dilate(mask, np.ones((k,k),np.uint8), iterations=1).astype(np.float32)
    if kernel == "min":
        return cv2.erode(mask, np.ones((k,k),np.uint8), iterations=1).astype(np.float32)
    if kernel == "gaussian":
        return cv2.GaussianBlur(mask.astype(np.float32), (k,k), sigmaX=1.0)
    if kernel == "bilateral":
        if image is None:
            raise ValueError("bilateral kernel requires image")
        # Use image-guided smoothing on mask values
        # Bilateral expects 8-bit image; we rescale image to [0,255]
        img8 = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # Apply bilateral on soft mask (float)
        m = (mask.astype(np.float32) * 255.0).astype(np.uint8)
        out = cv2.bilateralFilter(m, d=5, sigmaColor=50, sigmaSpace=50).astype(np.float32) / 255.0
        return out
    if kernel == "mode":
        # slow but acceptable for 224x224; for large, prefer mean/gaussian
        pad = 1
        m = np.pad(mask, pad, mode="edge")
        out = np.zeros_like(mask, dtype=np.float32)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                out[i,j] = _local_mode(m[i:i+3, j:j+3])
        return out
    raise ValueError(f"Unknown kernel: {kernel}")

@dataclass
class SAMConfig:
    morph: MorphName = "gradient"
    kernel: KernelName = "mean"
    alpha: float = 0.1  # weight on LSAM term

def build_adaptive_target(labels: torch.Tensor, images: Optional[torch.Tensor], num_classes: int, cfg: SAMConfig) -> torch.Tensor:
    """Build \tilde{y}_{AM} for each class, per-sample.
    labels: (N, H, W) or (N, D, H, W) int64
    images: (N, C, H, W) or (N, C, D, H, W), used only for bilateral guidance
    Returns: (N, C, *spatial) float32 in [0,1]
    """
    device = labels.device
    labels_np = labels.detach().cpu().numpy()
    imgs_np = None
    if images is not None:
        imgs_np = images.detach().cpu().numpy()
    out = []
    for n in range(labels_np.shape[0]):
        lab = labels_np[n]
        # Use first channel for guidance if needed
        img_guidance = None
        if imgs_np is not None:
            img_guidance = imgs_np[n,0]  # (H,W) or (D,H,W)
        if lab.ndim == 2:
            c_maps = []
            for c in range(num_classes):
                bin_mask = (lab == c).astype(np.uint8)
                morph_mask = apply_morphology(bin_mask, cfg.morph)
                guide = img_guidance if cfg.kernel=="bilateral" else None
                am = aggregate_kernel(morph_mask, cfg.kernel, image=guide)
                c_maps.append(am)
            out.append(np.stack(c_maps, axis=0))
        elif lab.ndim == 3:
            # slice-wise 2D operations
            D,H,W = lab.shape
            c_maps = []
            for c in range(num_classes):
                am_slices = []
                for d in range(D):
                    bin_mask = (lab[d] == c).astype(np.uint8)
                    morph_mask = apply_morphology(bin_mask, cfg.morph)
                    guide = img_guidance[d] if (cfg.kernel=="bilateral" and img_guidance is not None) else None
                    am = aggregate_kernel(morph_mask, cfg.kernel, image=guide)
                    am_slices.append(am)
                c_maps.append(np.stack(am_slices, axis=0))
            out.append(np.stack(c_maps, axis=0))
        else:
            raise ValueError(f"Unsupported label ndim: {lab.ndim}")
    tgt = torch.from_numpy(np.stack(out, axis=0)).float().to(device)
    # normalize to simplex-ish; avoid div by zero
    denom = tgt.sum(dim=1, keepdim=True).clamp_min(1e-6)
    tgt = tgt / denom
    return tgt
