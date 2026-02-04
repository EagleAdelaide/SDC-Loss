from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import load_yaml
from .data import NpzSegDataset, collate_samples
from .models import UNet2D, UNet3D
from .utils import set_seed, get_device, AverageMeter
from . import losses as L

LOSS_REGISTRY = {
    "dicece": "DiceCE",
    "fl": "Focal",
    "ecp": "ECP",
    "ls": "LabelSmoothing",
    "svls": "SVLS",
    "mbls": "MbLS",
    "nacl": "NACL",
    "fcl": "FCL",
    "sdc": "SDC",
}

def build_model(cfg: Dict[str, Any]) -> torch.nn.Module:
    m = cfg["model"]
    arch = m.get("arch", "unet2d").lower()
    in_ch = int(m.get("in_channels", 1))
    num_classes = int(m.get("num_classes", 2))
    base = int(m.get("base", 32 if arch=="unet2d" else 16))
    if arch == "unet2d":
        return UNet2D(in_channels=in_ch, num_classes=num_classes, base=base)
    if arch == "unet3d":
        return UNet3D(in_channels=in_ch, num_classes=num_classes, base=base)
    raise ValueError(f"Unsupported arch: {arch}")

def compute_loss(method: str, logits: torch.Tensor, labels: torch.Tensor, images: Optional[torch.Tensor], num_classes: int, cfg: Dict[str, Any]) -> torch.Tensor:
    method = method.lower()
    if method == "dicece":
        c = L.DiceCEConfig(**cfg.get("dicece", {}))
        return L.dice_ce_loss(logits, labels, num_classes, c)
    if method == "fl":
        c = L.FocalConfig(**cfg.get("focal", {}))
        return L.focal_loss(logits, labels, c)
    if method == "ecp":
        ce = torch.nn.functional.cross_entropy(logits, labels)
        c = L.ECPConfig(**cfg.get("ecp", {}))
        return ce + L.entropy_confidence_penalty(logits, c)
    if method == "ls":
        c = L.LabelSmoothingConfig(**cfg.get("labelsmoothing", {}))
        return L.label_smoothing_ce(logits, labels, num_classes, c)
    if method == "svls":
        c = L.SVLSConfig(**cfg.get("svls", {}))
        return L.svls_loss(logits, labels, images, num_classes, c)
    if method == "mbls":
        c = L.MbLSConfig(**cfg.get("mbls", {}))
        return L.mbls_loss(logits, labels, num_classes, c)
    if method == "nacl":
        c = L.NACLConfig(**cfg.get("nacl", {}))
        return L.nacl_loss(logits, labels, images, num_classes, c)
    if method == "fcl":
        c = L.FCLConfig(**cfg.get("fcl", {}))
        return L.focal_calibration_loss(logits, labels, num_classes, c)
    if method == "sdc":
        c = L.SDCConfig(**cfg.get("sdc", {}))
        return L.sdc_loss(logits, labels, images, num_classes, c)
    raise ValueError(f"Unknown method: {method}")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--outdir", type=str, default="runs")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg.get("seed", 42)))
    device = get_device(args.device)

    ds_cfg = cfg["dataset"]
    root = ds_cfg["root"]
    train_ds = NpzSegDataset(root=root, split="train")
    val_ds = NpzSegDataset(root=root, split="val")
    batch_size = int(cfg.get("batch_size", 4))
    num_workers = int(cfg.get("num_workers", 2))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_samples)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_samples)

    model = build_model(cfg).to(device)
    num_classes = int(cfg["model"]["num_classes"])

    opt_cfg = cfg.get("optim", {})
    lr = float(opt_cfg.get("lr", 1e-4))
    wd = float(opt_cfg.get("weight_decay", 1e-2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    sch_cfg = cfg.get("scheduler", {"type":"cosine", "epochs": int(cfg.get("epochs", 50))})
    epochs = int(cfg.get("epochs", 50))
    if sch_cfg.get("type","cosine") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        scheduler = None

    run_name = cfg.get("run_name", Path(args.config).stem)
    outdir = Path(args.outdir) / run_name
    outdir.mkdir(parents=True, exist_ok=True)

    method = cfg["method"].lower()
    best_val = float("inf")

    for epoch in range(1, epochs+1):
        model.train()
        meter = AverageMeter()
        pbar = tqdm(train_loader, desc=f"train {epoch}/{epochs}")
        for images, labels, _ in pbar:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = compute_loss(method, logits, labels, images, num_classes, cfg)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            meter.update(loss.item(), n=images.size(0))
            pbar.set_postfix(loss=meter.avg, lr=optimizer.param_groups[0]["lr"])
        if scheduler is not None:
            scheduler.step()

        # simple val loss
        model.eval()
        v_meter = AverageMeter()
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
                loss = compute_loss(method, logits, labels, images, num_classes, cfg)
                v_meter.update(loss.item(), n=1)
        torch.save({"model": model.state_dict(), "cfg": cfg}, outdir/"last.pt")
        if v_meter.avg < best_val:
            best_val = v_meter.avg
            torch.save({"model": model.state_dict(), "cfg": cfg}, outdir/"best.pt")
        with (outdir/"log.txt").open("a", encoding="utf-8") as f:
            f.write(f"epoch={epoch} train_loss={meter.avg:.6f} val_loss={v_meter.avg:.6f} best_val={best_val:.6f}\n")

    print(f"Saved checkpoints to {outdir}")

if __name__ == "__main__":
    main()
