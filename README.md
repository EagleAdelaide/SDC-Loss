# SDC Reproduction

This repo is an implementation for:

> **Calibrating on Medical Segmentation Model through Signed-Distance** (CIKM 2025)  
<img width="4390" height="2621" alt="image" src="https://github.com/user-attachments/assets/86b16ed1-84ed-4af1-bb82-bd91d37791ef" />
<img width="5564" height="1517" alt="image" src="https://github.com/user-attachments/assets/95fed592-3c46-4ad5-a117-31dff020fc5d" />


It includes:
- **SDC loss** = CE + *local calibration* (SAM) + *SDF alignment* (Listing 2)
- **SAM** (Spatially Adaptive Margin) module (morphology + local aggregation)
- **pECE** metric (Algorithm 1) with **false-positive offset**
- A unified training/evaluation pipeline for common baselines used in the paper:
  `DiceCE`, `Focal`, `ECP`, `Label Smoothing`, `SVLS` (approx), `MbLS` (lightweight), `NACL` (approx), `FCL`, `SDC`

> ⚠️ Note on baselines: for **NACL/SVLS/MbLS**, please refer to the original authors’ full code (some papers have additional details).
> 
> NACL: https://github.com/Bala93/MarginLoss
> 
> SVLS: https://github.com/mobarakol/SVLS
> 
> MbLS: https://github.com/by-liu/MbLS/
> 
> If you point the pipeline to the **exact same data split + preprocessing + training schedule**, you should be able to reproduce the tables. If you want bitwise-identical numbers, use the original baseline repos and plug their predictions into `sdc-eval`.

---

## 0) Environment

Tested with Python ≥ 3.9.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Dependencies (declared in `pyproject.toml`): `torch`, `opencv-python`, `numpy`, `scipy`, `pyyaml`, `pandas`, `tqdm`.

---

## 1) Quick Smoke Test (toy dataset)

This runs end-to-end in minutes and produces a small results table.

### 1.1 Generate toy data
```bash
python scripts/make_toy_data.py --out data/toy
```

### 1.2 Train a method (example: SDC)
```bash
python -m sdc.train --config configs/toy_sdc.yaml --outdir runs
```

### 1.3 Inference + evaluation
```bash
python -m sdc.infer --config configs/toy_sdc.yaml --ckpt runs/toy_sdc/best.pt --split test --outdir preds
python -m sdc.eval  --pred_root preds --num_classes 2 --out_csv toy_results.csv
```

You should see a printed table and `toy_results.csv`.

---

## 2) Reproducing Tables (ACDC / FLARE / BraTS / Prostate)

### 2.1 Required data format

This repo expects each dataset in the following lightweight format:

```
DATA_ROOT/
  train/*.npz
  val/*.npz
  test/*.npz
```

Each `.npz` must contain:
- `image`: float32, shape `(C,H,W)` for 2D or `(C,D,H,W)` for 3D
- `label`: int64, shape `(H,W)` or `(D,H,W)` with class indices

> If your data is in NIfTI (`.nii.gz`), you can write a one-time converter that:
> 1) resamples/crops to the paper’s resolution, and  
> 2) stores arrays as `.npz` for fast training.

For FLARE: the paper mentions volumes resampled and cropped to **192×192×30**.

### 2.2 Training settings 

- Optimizer: **AdamW**
- Initial LR: **1e-4**
- Scheduler: **cosine decay**
- Batch size: **4**
- Local calibration window: **k=3** (Eq. (6))
- Ground-truth SDFs: precomputed via `scipy.ndimage.distance_transform_edt`
- Morphological ops: OpenCV `cv2` (3×3 structuring element)
- pECE: `bins=10`, `fp_weight=2.0` (Algorithm 1)

All of these are configurable in the YAML configs.

### 2.3 Train / infer / eval for a dataset

1) Create a dataset config:
```yaml
# example: configs/unet_acdc_sdc.yaml
seed: 42
run_name: unet_acdc_sdc
method: sdc

dataset:
  name: acdc
  root: /path/to/ACDC_NPZ

model:
  arch: unet2d
  in_channels: 1
  num_classes: 2
  base: 32

batch_size: 4
epochs: 200
optim:
  lr: 1.0e-4
  weight_decay: 1.0e-2

sdc:
  alpha: 0.1
  lam_sdf: 0.1
  sam_morph: gradient
  sam_kernel: mean
  sdf_scale: 100.0
```

2) Train:
```bash
python -m sdc.train --config configs/unet_acdc_sdc.yaml --outdir runs
```

3) Inference:
```bash
python -m sdc.infer --config configs/unet_acdc_sdc.yaml --ckpt runs/unet_acdc_sdc/best.pt --split test --outdir preds
```

4) Evaluate all runs under a predictions root:
```bash
python -m sdc.eval --pred_root preds --num_classes 2 --out_csv acdc_table.csv
```

`preds/` should contain one subfolder per method/run (each with `*.npz` predictions).

---

## 3) Methods Implemtation

### 3.1 SDC loss (Listing 2)

- (i) **Pixel fidelity**: `CE(seg_logits, targets)`
- (ii) **Local calibration**: `|utargets - softmax(seg_logits)|` (SAM target)
- (iii) **SDF alignment**: `L1( 2*probs-1 , sdf_target ) / 100`

Total:
```
loss = CE + alpha * LC + lam_sdf * SDF
```

### 3.2 SAM (Spatially Adaptive Margin)

- Morphology `M_phi` ∈ {erosion, dilation, opening, closing, internal boundary, external boundary, gradient}
- Local aggregation kernel ∈ {mean, max, min, mode, gaussian, bilateral}

Produces an adaptive soft target map `y_AM` used as local calibration supervision.

### 3.3 pECE (Algorithm 1)

We implement the binning algorithm with a **false-positive confidence offset**:
- Confidence `p` is treated as **foreground confidence** `p_fg = 1 - p_bg`
- Ground-truth `y` is the **foreground indicator** `(label != background)`
- False positives are `y==0` with high confidence and get extra penalty via `fp_weight`.

---

## 4) Citation

If you use this code, please cite the paper:

```bibtex
@inproceedings{liang2025sdc,
  title={Calibrating on Medical Segmentation Model through Signed-Distance},
  author={Liang, Wenhao and Zhang, Emma Wei and Yue, Lin and Xu, Miao and Maennel, Olaf and Chen, Weitong},
  booktitle={Proceedings of the 34th ACM International Conference on Information and Knowledge Management (CIKM)},
  year={2025}
}
```

---

## 5) Repo structure

- `src/sdc/` core library
  - `models.py` U-Net (2D/3D)
  - `losses.py` SDC + baselines
  - `sam.py` SAM module (morphology + aggregation)
  - `sdf.py` signed distance targets
  - `metrics.py` DSC / HD95 / ECE / CECE / pECE
- `configs/` YAML training configs (toy examples included)
- `scripts/` utilities (toy data generator)

---

## Troubleshooting

- **OpenCV install issues**: try `pip install opencv-python-headless`
- **CUDA OOM** (3D): reduce `model.base`, crop volumes, or use fewer classes
- **HD95 = inf**: usually means empty prediction or empty GT for foreground in a case
