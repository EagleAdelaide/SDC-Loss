from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal, Dict

import torch
import torch.nn.functional as F

from .utils import one_hot
from .sam import SAMConfig, build_adaptive_target
from .sdf import multiclass_sdf_targets

def soft_dice_loss(probs: torch.Tensor, targets_1h: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # probs, targets: (N,C,*spatial)
    dims = tuple(range(2, probs.ndim))
    inter = (probs * targets_1h).sum(dims)
    denom = (probs + targets_1h).sum(dims).clamp_min(eps)
    dice = (2*inter + eps) / (denom + eps)
    return 1.0 - dice.mean()

@dataclass
class DiceCEConfig:
    dice_weight: float = 1.0
    ce_weight: float = 1.0

def dice_ce_loss(logits: torch.Tensor, labels: torch.Tensor, num_classes: int, cfg: DiceCEConfig) -> torch.Tensor:
    ce = F.cross_entropy(logits, labels)
    probs = F.softmax(logits, dim=1)
    tgt = one_hot(labels, num_classes)
    dice = soft_dice_loss(probs, tgt)
    return cfg.ce_weight * ce + cfg.dice_weight * dice

@dataclass
class FocalConfig:
    gamma: float = 3.0

def focal_loss(logits: torch.Tensor, labels: torch.Tensor, cfg: FocalConfig) -> torch.Tensor:
    # Multiclass focal loss using CE with modulating factor.
    logp = F.log_softmax(logits, dim=1)
    p = torch.exp(logp)
    # gather true class probability
    y = labels.long()
    pt = p.gather(1, y.unsqueeze(1)).squeeze(1)
    logpt = logp.gather(1, y.unsqueeze(1)).squeeze(1)
    loss = -((1-pt) ** cfg.gamma) * logpt
    return loss.mean()

@dataclass
class LabelSmoothingConfig:
    alpha: float = 0.1

def label_smoothing_ce(logits: torch.Tensor, labels: torch.Tensor, num_classes: int, cfg: LabelSmoothingConfig) -> torch.Tensor:
    # standard LS on CE
    with torch.no_grad():
        true_dist = torch.zeros_like(logits)
        true_dist.fill_(cfg.alpha / (num_classes - 1))
        true_dist.scatter_(1, labels.unsqueeze(1), 1.0 - cfg.alpha)
    logp = F.log_softmax(logits, dim=1)
    return -(true_dist * logp).sum(dim=1).mean()

@dataclass
class ECPConfig:
    lam: float = 0.1

def entropy_confidence_penalty(logits: torch.Tensor, cfg: ECPConfig) -> torch.Tensor:
    # Penalize low entropy -> encourages softer predictions; common form: -H(p)
    p = F.softmax(logits, dim=1).clamp_min(1e-8)
    ent = -(p * p.log()).sum(dim=1)  # (N,*)
    return -cfg.lam * ent.mean()

@dataclass
class FCLConfig:
    gamma: float = 3.0
    lam: float = 0.1

def brier_score(probs: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    tgt = one_hot(labels, num_classes)
    return ((probs - tgt) ** 2).mean()

def focal_calibration_loss(logits: torch.Tensor, labels: torch.Tensor, num_classes: int, cfg: FCLConfig) -> torch.Tensor:
    fl = focal_loss(logits, labels, FocalConfig(gamma=cfg.gamma))
    probs = F.softmax(logits, dim=1)
    brier = brier_score(probs, labels, num_classes)
    return fl + cfg.lam * brier

@dataclass
class SVLSConfig:
    sigma: float = 2.0
    alpha: float = 0.1
    dist: Literal["l1","l2"] = "l1"

def svls_loss(logits: torch.Tensor, labels: torch.Tensor, images: Optional[torch.Tensor], num_classes: int, cfg: SVLSConfig) -> torch.Tensor:
    # Approximate SVLS by using SAM with gaussian kernel on the raw label mask (no morphology)
    sam_cfg = SAMConfig(morph="closing", kernel="gaussian", alpha=cfg.alpha)  # morph is irrelevant-ish for closing; keeps mask stable
    y_am = build_adaptive_target(labels, images, num_classes, sam_cfg)
    probs = F.softmax(logits, dim=1)
    ce = F.cross_entropy(logits, labels)
    if cfg.dist == "l2":
        d = (probs - y_am).pow(2).mean()
    else:
        d = (probs - y_am).abs().mean()
    return ce + cfg.alpha * d

@dataclass
class NACLConfig:
    kernel: Literal["mean","max","min","mode","gaussian","bilateral"] = "min"
    alpha: float = 0.1
    dist: Literal["l1","l2"] = "l1"

def nacl_loss(logits: torch.Tensor, labels: torch.Tensor, images: Optional[torch.Tensor], num_classes: int, cfg: NACLConfig) -> torch.Tensor:
    # Neighbor-Aware Calibration Loss (approx): CE + alpha * D(p, y_am) where y_am is neighbor frequency map without morphology.
    sam_cfg = SAMConfig(morph="closing", kernel=cfg.kernel, alpha=cfg.alpha)
    y_am = build_adaptive_target(labels, images, num_classes, sam_cfg)
    probs = F.softmax(logits, dim=1)
    ce = F.cross_entropy(logits, labels)
    if cfg.dist == "l2":
        d = (probs - y_am).pow(2).mean()
    else:
        d = (probs - y_am).abs().mean()
    return ce + cfg.alpha * d

@dataclass
class MbLSConfig:
    m: float = 10.0
    alpha: float = 0.1

def mbls_loss(logits: torch.Tensor, labels: torch.Tensor, num_classes: int, cfg: MbLSConfig) -> torch.Tensor:
    # Lightweight approximation: encourage margin between top1 and runner-up logits, combined with LS.
    ls = label_smoothing_ce(logits, labels, num_classes, LabelSmoothingConfig(alpha=cfg.alpha))
    top2 = torch.topk(logits, k=2, dim=1).values
    margin = (top2[:,0] - top2[:,1])
    # penalize too small margin
    margin_pen = F.relu(cfg.m - margin).mean() * 1e-3
    return ls + margin_pen

@dataclass
class SDCConfig:
    alpha: float = 0.1     # weight on local calibration (SAM)
    lam_sdf: float = 0.1   # lambda_sdf
    sam_morph: str = "gradient"
    sam_kernel: str = "mean"
    sdf_scale: float = 100.0
    dist: Literal["l1","l2"] = "l1"

def sdc_loss(logits: torch.Tensor, labels: torch.Tensor, images: Optional[torch.Tensor], num_classes: int, cfg: SDCConfig) -> torch.Tensor:
    # (i) pixel fidelity
    loss_ce = F.cross_entropy(logits, labels)
    probs = F.softmax(logits, dim=1)
    # (ii) local calibration via SAM target
    sam_cfg = SAMConfig(morph=cfg.sam_morph, kernel=cfg.sam_kernel, alpha=cfg.alpha)
    utargets = build_adaptive_target(labels, images, num_classes, sam_cfg)
    loss_lc = (utargets - probs).abs().mean() if cfg.dist=="l1" else (utargets - probs).pow(2).mean()
    # (iii) SDF alignment (Listing 2)
    sdf_tgt = multiclass_sdf_targets(labels, num_classes, normalize=True)
    sdf_pre = 2.0 * probs - 1.0
    if cfg.dist == "l1":
        loss_sdf = F.l1_loss(sdf_pre, sdf_tgt) / cfg.sdf_scale
    else:
        loss_sdf = F.mse_loss(sdf_pre, sdf_tgt) / cfg.sdf_scale
    return loss_ce + cfg.alpha * loss_lc + cfg.lam_sdf * loss_sdf
