from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

@dataclass
class Sample:
    image: torch.Tensor   # (C, *spatial)
    label: torch.Tensor   # (*spatial), int64
    meta: Dict

class NpzSegDataset(Dataset):
    """A lightweight dataset that reads .npz files with keys: 'image', 'label'.

    - image: float32 array shaped (C,H,W) for 2D or (C,D,H,W) for 3D
    - label: int64 array shaped (H,W) or (D,H,W)
    """
    def __init__(self, root: str | Path, split: str = "train") -> None:
        self.root = Path(root)
        self.split = split
        self.files = sorted((self.root / split).glob("*.npz"))
        if len(self.files) == 0:
            raise FileNotFoundError(f"No .npz files found under {self.root/split}. Expected structure: root/train/*.npz, root/val/*.npz, root/test/*.npz")
    def __len__(self) -> int:
        return len(self.files)
    def __getitem__(self, idx: int) -> Sample:
        path = self.files[idx]
        arr = np.load(path)
        image = torch.from_numpy(arr["image"]).float()
        label = torch.from_numpy(arr["label"]).long()
        meta = {"id": path.stem, "path": str(path)}
        return Sample(image=image, label=label, meta=meta)

def collate_samples(batch: List[Sample]) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
    images = torch.stack([b.image for b in batch], dim=0)
    labels = torch.stack([b.label for b in batch], dim=0)
    metas = [b.meta for b in batch]
    return images, labels, metas
