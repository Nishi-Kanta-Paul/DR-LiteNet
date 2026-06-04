# DR-LiteNet

**DR-LiteNet: A Lightweight Explainable Hybrid CNN Framework for Imbalanced Diabetic Retinopathy Grading Using Adaptive SMOTE Fusion**

---

## Overview

DR-LiteNet is a dual-branch hybrid convolutional neural network for automated five-class diabetic retinopathy (DR) severity grading from colour fundus photographs. It fuses lightweight pretrained CNN deep features (EfficientNetB0 or MobileNetV3-Small) with four handcrafted retinal lesion descriptors, applies **adaptive SMOTE at the fused feature level** to correct class imbalance, and provides post-hoc **Grad-CAM** explainability — all within Google Colab free-tier GPU constraints (< 10M parameters).

### Architecture

```
Input fundus image (224×224×3)
        │
        ├─────────────────────┐
        ▼                     ▼
  CNN Branch              Lesion Branch
  EfficientNetB0          ─ Vessel density (1-d)
  (pretrained, GAP)       ─ Exudate intensity (1-d)
  → 1280-d                ─ Texture entropy (1-d)
                          ─ HSV histogram (48-d)
                          → 52-d
        │                     │
        └──────┬──────────────┘
               ▼
        Feature Fusion (concat) → 1332-d
               │
        [Training only]
        Adaptive SMOTE (ADASYN)
               │
               ▼
        Dense Head
        Linear(1332→512) → ReLU → Dropout(0.4) → Linear(512→5)
               │
               ▼
        5-class DR severity (softmax)
```

### Key Contributions

- Feature-level Adaptive SMOTE (ADASYN) for minority-class DR grading
- Dual-branch hybrid fusion: CNN deep features + handcrafted lesion descriptors
- Lightweight backbone (< 5M params) feasible on Colab free-tier GPU
- Grad-CAM saliency maps for clinical explainability (RQ4)
- Systematic 5-fold stratified cross-validation on APTOS 2019

---

## Installation

```bash
git clone <repo_url>
cd DRLiteNet
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, PyTorch 2.3+, CUDA GPU recommended (T4/P100 on Colab).

---

## Data Preparation

### APTOS 2019 (Primary)

Download from [Kaggle APTOS 2019](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data) and place as:

```
data/
├── train_1.csv              # columns: id_code, diagnosis  (2929 images)
├── valid.csv                # columns: id_code, diagnosis  (365 images)
├── test.csv                 # columns: id_code, diagnosis  (868 images)
├── train_images/
│   └── train_images/        # *.png fundus images
├── val_images/
│   └── val_images/
└── test_images/
    └── test_images/
```

### IDRiD (External Validation — Optional)

Download from [IDRiD Challenge](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid) and place as:

```
data/
└── idrid/
    ├── images/              # *.jpg fundus images
    └── labels.csv           # columns: id_code, diagnosis (0–4 scale)
```

---

## Usage

### Training

```bash
# Train DR-LiteNet (EfficientNetB0), all 5 folds:
python src/main.py --mode train --model dr_litenet_effb0

# Train MobileNetV3-Small variant:
python src/main.py --mode train --model dr_litenet_mobv3

# Quick sanity run (2 epochs, fold 0):
python src/main.py --mode train --model dr_litenet_effb0 \
       --fold 0 --p1_epochs 2 --p2_epochs 1 --batch_size 8

# Train baselines:
python baselines/train_effb0_baseline.py --all_folds
python baselines/train_resnet50_baseline.py --all_folds
# ... (see baselines/ for all scripts)

# Full pipeline (train + eval + compare + explain):
bash scripts/train.sh
```

### Evaluation

```bash
python src/main.py --mode evaluate --model dr_litenet_effb0
# Saves: outputs/tables/, outputs/figures/
```

### Inference

```bash
# Single image:
python src/main.py --mode infer --model dr_litenet_effb0 \
       --input path/to/fundus.jpg --gradcam

# Directory:
python src/main.py --mode infer --model dr_litenet_effb0 \
       --input data/val_images/val_images/

# IDRiD external validation:
python src/main.py --mode infer --model dr_litenet_effb0 --idrid
```

### Explainability (Grad-CAM)

```bash
python src/main.py --mode explain --model dr_litenet_effb0 --fold 0
# Saves: outputs/figures/grad_cam/
```

### Comparison Tables

```bash
python src/main.py --mode compare
# Saves: outputs/tables/baseline_comparison.csv, ablation_study.csv
```

---

## Project Structure

```
DRLiteNet/
├── src/
│   ├── config.py           # All hyperparameters and paths
│   ├── dataset.py          # Data loading, preprocessing, handcrafted features
│   ├── model.py            # DR-LiteNet architecture
│   ├── train.py            # Two-phase training pipeline
│   ├── evaluate.py         # Metrics, plots, tables
│   ├── explainability.py   # Grad-CAM
│   ├── inference.py        # Inference CLI
│   ├── utils.py            # Shared helpers
│   └── main.py             # Unified CLI entry point
├── baselines/
│   ├── _baseline_engine.py
│   ├── train_effb0_baseline.py
│   ├── train_effb0_weighted.py
│   ├── train_mobv3_baseline.py
│   ├── train_resnet50_baseline.py
│   ├── train_dr_litenet_no_smote.py
│   ├── train_dr_litenet_weighted.py
│   └── compare_results.py
├── data/                   # APTOS 2019 + IDRiD (not tracked by git)
├── experiments/            # Checkpoints + logs (per model, per fold)
├── outputs/
│   ├── figures/            # Plots, Grad-CAM heatmaps
│   ├── tables/             # CSV metric tables
│   └── predictions/        # Inference outputs
├── notebooks/              # Exploratory notebooks
├── scripts/
│   └── train.sh            # Full pipeline script
├── requirements.txt
└── README.md
```

---

## Results

*Results below are placeholders — to be filled after full GPU training on the complete APTOS 2019 dataset.*

### DR-LiteNet (EfficientNetB0) — 5-Fold Cross-Validation on APTOS 2019

| Metric | Mean ± Std |
|---|---|
| Accuracy | — |
| QWK | — |
| Macro F1 | — |
| Severe Sensitivity | — |
| Proliferative DR Sensitivity | — |

### Baseline Comparison

| Model | Accuracy | QWK | Macro F1 |
|---|---|---|---|
| DR-LiteNet (EfficientNetB0) | — | — | — |
| DR-LiteNet (MobileNetV3) | — | — | — |
| EfficientNetB0 baseline | — | — | — |
| EfficientNetB0 + class weights | — | — | — |
| MobileNetV3-Small | — | — | — |
| ResNet50 | — | — | — |
| DR-LiteNet (no SMOTE) | — | — | — |
| DR-LiteNet (class-weighted) | — | — | — |

### Target Performance

- Overall Accuracy > 90%
- Quadratic Weighted Kappa (QWK) > 0.85
- Parameter count < 10M (EfficientNetB0: 4.69M, MobileNetV3: 1.25M)

---

## Citation

```bibtex
@article{drlitenet2025,
  title   = {DR-LiteNet: A Lightweight Explainable Hybrid CNN Framework for
             Imbalanced Diabetic Retinopathy Grading Using Adaptive SMOTE Fusion},
  author  = {[Author]},
  journal = {[Journal]},
  year    = {2026},
}
```

---

## License

[To be determined]
