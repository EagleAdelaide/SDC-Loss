#!/usr/bin/env python3
"""Generate a tiny synthetic segmentation dataset (2D circles) for smoke tests.

Output: data/toy/{train,val,test}/*.npz with keys: image (1,H,W), label (H,W).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

def make_sample(rng: np.random.Generator, H: int, W: int) -> tuple[np.ndarray, np.ndarray]:
    img = rng.normal(0, 1, size=(H,W)).astype(np.float32)
    label = np.zeros((H,W), dtype=np.int64)

    r = int(rng.integers(low=10, high=max(11, min(H,W)//4)))
    cy = int(rng.integers(low=r, high=H-r))
    cx = int(rng.integers(low=r, high=W-r))

    yy, xx = np.ogrid[:H, :W]
    mask = (yy - cy)**2 + (xx - cx)**2 <= r**2
    label[mask] = 1

    img = img + 2.0 * label.astype(np.float32)
    # simple blur via separable box filter
    k = 5
    ker = np.ones(k, dtype=np.float32) / k
    img = np.apply_along_axis(lambda m: np.convolve(m, ker, mode="same"), 0, img)
    img = np.apply_along_axis(lambda m: np.convolve(m, ker, mode="same"), 1, img)
    img = (img - img.mean()) / (img.std() + 1e-6)
    return img[None,...], label

def write_split(out_root: Path, split: str, n: int, seed: int, H: int, W: int) -> None:
    rng = np.random.default_rng(seed)
    (out_root/split).mkdir(parents=True, exist_ok=True)
    for i in range(n):
        img, lab = make_sample(rng, H, W)
        np.savez_compressed(out_root/split/f"{split}_{i:03d}.npz", image=img, label=lab)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data/toy")
    ap.add_argument("--H", type=int, default=128)
    ap.add_argument("--W", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_train", type=int, default=64)
    ap.add_argument("--n_val", type=int, default=16)
    ap.add_argument("--n_test", type=int, default=16)
    args = ap.parse_args()

    out_root = Path(args.out)
    write_split(out_root, "train", args.n_train, args.seed, args.H, args.W)
    write_split(out_root, "val", args.n_val, args.seed+1, args.H, args.W)
    write_split(out_root, "test", args.n_test, args.seed+2, args.H, args.W)
    print(f"Wrote toy dataset to {out_root}")

if __name__ == "__main__":
    main()
