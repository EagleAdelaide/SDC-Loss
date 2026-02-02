from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import load_yaml
from .data import NpzSegDataset, collate_samples
from .models import UNet2D, UNet3D
from .utils import get_device

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

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--split", type=str, default="test", choices=["train","val","test"])
    ap.add_argument("--outdir", type=str, default="preds")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    device = get_device(args.device)

    ds_cfg = cfg["dataset"]
    root = ds_cfg["root"]
    ds = NpzSegDataset(root=root, split=args.split)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_samples)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = build_model(cfg)
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()

    outdir = Path(args.outdir) / Path(args.ckpt).parent.name
    outdir.mkdir(parents=True, exist_ok=True)

    for images, labels, metas in tqdm(loader, desc=f"infer {args.split}"):
        images = images.to(device)
        with torch.no_grad():
            logits = model(images)
            probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]  # (C,*)
        lab = labels.numpy()[0]
        sid = metas[0]["id"]
        np.savez_compressed(outdir/f"{sid}.npz", probs=probs.astype(np.float32), label=lab.astype(np.int64))

    print(f"Saved predictions to {outdir}")

if __name__ == "__main__":
    main()
