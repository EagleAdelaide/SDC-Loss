from __future__ import annotations
import os, random
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

@dataclass
class AverageMeter:
    total: float = 0.0
    count: int = 0
    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * int(n)
        self.count += int(n)
    @property
    def avg(self) -> float:
        return self.total / max(1, self.count)

def get_device(device: Optional[str] = None) -> torch.device:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)

def one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    # labels: (N, *spatial), int64
    return torch.nn.functional.one_hot(labels.long(), num_classes=num_classes).permute(0, -1, *range(1, labels.ndim)).float()
