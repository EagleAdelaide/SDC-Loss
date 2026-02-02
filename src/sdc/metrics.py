from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_erosion, distance_transform_edt

from .utils import to_numpy

def dice_score(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    return float((2*inter + eps) / (denom + eps))

def hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return float("inf")
    # surface extraction
    struct = np.ones((3,)*pred.ndim, dtype=bool)
    pred_er = binary_erosion(pred, structure=struct, border_value=0)
    gt_er = binary_erosion(gt, structure=struct, border_value=0)
    pred_surf = np.logical_xor(pred, pred_er)
    gt_surf = np.logical_xor(gt, gt_er)
    # distance to surface
    dt_gt = distance_transform_edt(~gt_surf)
    dt_pred = distance_transform_edt(~pred_surf)
    d1 = dt_gt[pred_surf]
    d2 = dt_pred[gt_surf]
    all_d = np.concatenate([d1, d2]).astype(np.float32)
    return float(np.percentile(all_d, 95))

@dataclass
class CalibConfig:
    bins: int = 10
    prob_threshold: float = 1e-3
    ignore_background: bool = False
    fp_weight: float = 2.0

def ece_binary(conf: np.ndarray, y_true: np.ndarray, cfg: CalibConfig) -> float:
    # conf in [0,1], y_true in {0,1}
    conf = conf.reshape(-1)
    y_true = y_true.reshape(-1)
    edges = np.linspace(0.0, 1.0, cfg.bins + 1)
    ece = 0.0
    n = conf.size
    for i in range(cfg.bins):
        lo, hi = edges[i], edges[i+1]
        m = (conf > lo) & (conf <= hi)
        if not np.any(m):
            continue
        p_bar = conf[m].mean()
        a_bar = y_true[m].mean()
        ece += (m.sum()/n) * abs(p_bar - a_bar)
    return float(ece)

def ece_multiclass(probs: np.ndarray, labels: np.ndarray, cfg: CalibConfig) -> float:
    # Eq.(3) foreground ECE: we compute on foreground confidence (1 - p_bg) and y_fg (gt != bg)
    if probs.ndim < 2:
        raise ValueError("probs must be (C,*spatial)")
    bg = 0
    p_fg = 1.0 - probs[bg]
    y_fg = (labels != bg).astype(np.float32)
    return ece_binary(p_fg, y_fg, cfg)

def cece(probs: np.ndarray, labels: np.ndarray, cfg: CalibConfig) -> float:
    # Eq.(4): average ECE per class; optionally ignore background.
    C = probs.shape[0]
    eces = []
    for c in range(C):
        if cfg.ignore_background and c == 0:
            continue
        pc = probs[c]
        yc = (labels == c).astype(np.float32)
        # optionally discard negligible representation by threshold on predicted prob
        m = (pc > cfg.prob_threshold)
        if m.sum() == 0:
            continue
        eces.append(ece_binary(pc[m], yc[m], cfg))
    if len(eces) == 0:
        return 0.0
    return float(np.mean(eces))

def pECE(probs: np.ndarray, labels: np.ndarray, cfg: CalibConfig) -> float:
    """Algorithm 1 / Eq.(18) implementation (multi-class -> foreground confidence).
    - p: predicted foreground confidence map
    - y: ground-truth foreground indicator
    False positives = y==0 with high confidence.
    """
    bg = 0
    p = 1.0 - probs[bg]
    y = (labels != bg).astype(np.float32)
    p = p.reshape(-1)
    y = y.reshape(-1)
    edges = np.linspace(0.0, 1.0, cfg.bins + 1)
    total = p.size
    out = 0.0
    for i in range(cfg.bins):
        lo, hi = edges[i], edges[i+1]
        m = (p > lo) & (p <= hi)
        eta = int(m.sum())
        if eta == 0:
            continue
        bp = float(p[m].mean())
        ba = float(y[m].mean())
        fp_mask = m & (y == 0)
        bp_fp = float(p[fp_mask].mean()) if fp_mask.sum() > 0 else 0.0
        offset = cfg.fp_weight * bp_fp
        out += (eta / total) * abs((bp - ba) + offset)
    return float(out)

def compute_all_metrics(
    probs: np.ndarray, labels: np.ndarray, num_classes: int, cfg: CalibConfig
) -> Dict[str, float]:
    # probs: (C,*spatial), labels: (*spatial)
    # Dice & HD are computed for foreground union (non-background)
    pred_lbl = probs.argmax(axis=0)
    pred_fg = pred_lbl != 0
    gt_fg = labels != 0
    dsc = dice_score(pred_fg, gt_fg)
    h = hd95(pred_fg, gt_fg)
    return {
        "DSC": dsc,
        "HD95": h,
        "ECE": ece_multiclass(probs, labels, cfg),
        "CECE": cece(probs, labels, cfg),
        "pECE": pECE(probs, labels, cfg),
    }
