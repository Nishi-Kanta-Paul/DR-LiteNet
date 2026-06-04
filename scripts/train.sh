#!/bin/bash
# =============================================================================
# DR-LiteNet — Full Experimental Pipeline Reproduction Script
#
# Usage:
#   bash scripts/train.sh                  # full pipeline
#   bash scripts/train.sh --quick          # 2 epochs, fold 0 only (sanity check)
#
# Requirements: Python 3.10+, CUDA GPU recommended.
# Data: APTOS 2019 at data/{train_images,val_images,test_images}/ + CSVs.
#       IDRiD at data/idrid/ (optional — skip if not available).
# =============================================================================

set -euo pipefail

QUICK=0
for arg in "$@"; do
  [[ "$arg" == "--quick" ]] && QUICK=1
done

if [[ $QUICK -eq 1 ]]; then
  EPOCHS_P1=2; EPOCHS_P2=1; BATCH=8
  echo "[train.sh] QUICK MODE: 2 epochs, batch_size=8"
else
  EPOCHS_P1=20; EPOCHS_P2=30; BATCH=32
  echo "[train.sh] FULL MODE: p1=${EPOCHS_P1} p2=${EPOCHS_P2} epochs, batch_size=${BATCH}"
fi

echo "============================================================"
echo " Step 1: Install dependencies"
echo "============================================================"
pip install -r requirements.txt -q

echo "============================================================"
echo " Step 2: Train DR-LiteNet — EfficientNetB0 (all 5 folds)"
echo "============================================================"
python src/main.py --mode train --model dr_litenet_effb0 \
       --p1_epochs "$EPOCHS_P1" --p2_epochs "$EPOCHS_P2" --batch_size "$BATCH"

echo "============================================================"
echo " Step 3: Train DR-LiteNet — MobileNetV3-Small (all 5 folds)"
echo "============================================================"
python src/main.py --mode train --model dr_litenet_mobv3 \
       --p1_epochs "$EPOCHS_P1" --p2_epochs "$EPOCHS_P2" --batch_size "$BATCH"

echo "============================================================"
echo " Step 4: Train baseline models"
echo "============================================================"
for script in \
  baselines/train_effb0_baseline.py \
  baselines/train_effb0_weighted.py \
  baselines/train_mobv3_baseline.py \
  baselines/train_resnet50_baseline.py \
  baselines/train_dr_litenet_no_smote.py \
  baselines/train_dr_litenet_weighted.py
do
  echo "  Training $script ..."
  python "$script" --all_folds --epochs "$((EPOCHS_P1 + EPOCHS_P2))" --batch_size "$BATCH"
done

echo "============================================================"
echo " Step 5: Evaluate all models"
echo "============================================================"
python src/main.py --mode evaluate --model dr_litenet_effb0
python src/main.py --mode evaluate --model dr_litenet_mobv3

echo "============================================================"
echo " Step 6: Generate comparison / ablation tables"
echo "============================================================"
python src/main.py --mode compare

echo "============================================================"
echo " Step 7: Generate Grad-CAM visualisations"
echo "============================================================"
python src/main.py --mode explain --model dr_litenet_effb0 --fold 0

echo "============================================================"
echo " Step 8: Run inference on validation set"
echo "============================================================"
python src/main.py --mode infer --model dr_litenet_effb0 \
       --input data/val_images/val_images/

echo "============================================================"
echo " Step 9: IDRiD external validation (if data available)"
echo "============================================================"
if [[ -f data/idrid/labels.csv ]]; then
  python src/main.py --mode infer --model dr_litenet_effb0 --idrid
else
  echo "  [SKIP] IDRiD data not found at data/idrid/. Skipping external validation."
fi

echo "============================================================"
echo " Full pipeline complete."
echo " Results in:"
echo "   outputs/tables/       — CSV metric tables"
echo "   outputs/figures/      — plots and Grad-CAM"
echo "   outputs/predictions/  — inference JSON/CSV"
echo "   experiments/          — checkpoints and logs"
echo "============================================================"
