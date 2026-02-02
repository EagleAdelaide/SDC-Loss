from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd

from .metrics import compute_all_metrics, CalibConfig

def load_pred_folder(folder: str | Path) -> List[Path]:
    folder = Path(folder)
    files = sorted(folder.glob("*.npz"))
    if len(files) == 0:
        raise FileNotFoundError(f"No prediction files (*.npz) found in {folder}")
    return files

def evaluate_folder(folder: str | Path, calib_cfg: CalibConfig, num_classes: int) -> Dict[str, float]:
    files = load_pred_folder(folder)
    mets = []
    for f in files:
        arr = np.load(f)
        probs = arr["probs"]  # (C,*)
        label = arr["label"]
        mets.append(compute_all_metrics(probs, label, num_classes, calib_cfg))
    # mean over cases
    out = {k: float(np.mean([m[k] for m in mets])) for k in mets[0].keys()}
    return out

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_root", type=str, required=True, help="Root folder containing subfolders per method/run, each with *.npz")
    ap.add_argument("--num_classes", type=int, required=True)
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--fp_weight", type=float, default=2.0)
    ap.add_argument("--ignore_background", action="store_true")
    ap.add_argument("--out_csv", type=str, default="results.csv")
    args = ap.parse_args()

    calib_cfg = CalibConfig(bins=args.bins, fp_weight=args.fp_weight, ignore_background=args.ignore_background)

    pred_root = Path(args.pred_root)
    rows = []
    for sub in sorted([p for p in pred_root.iterdir() if p.is_dir()]):
        met = evaluate_folder(sub, calib_cfg, args.num_classes)
        met["run"] = sub.name
        rows.append(met)
    df = pd.DataFrame(rows).set_index("run").sort_values("DSC", ascending=False)
    df.to_csv(args.out_csv)
    print(df)
    print(f"Wrote {args.out_csv}")

if __name__ == "__main__":
    main()
