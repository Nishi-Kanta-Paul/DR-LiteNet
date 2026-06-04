"""
Baseline: MobileNetV3-Small — deep features only, no handcrafted, no SMOTE.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import src.config as cfg
from baselines._baseline_engine import BaselineModel, train_baseline_fold
from src.utils import get_device, set_seed

MODEL_NAME = "mobv3_baseline"
BACKBONE   = "mobilenet_v3_small"


def run(fold_idx: int = 0, num_epochs: int = cfg.MAX_EPOCHS,
        batch_size: int = cfg.BATCH_SIZE, num_workers: int = 2):
    set_seed(cfg.SEED)
    device = get_device()
    model  = BaselineModel(backbone_name=BACKBONE, pretrained=True).to(device)
    print(f"\n[{MODEL_NAME}] params={model.count_parameters(trainable_only=False):,}")
    return train_baseline_fold(
        fold_idx=fold_idx, model_name=MODEL_NAME, model=model, device=device,
        use_class_weights=False, use_handcrafted=False,
        num_epochs=num_epochs, batch_size=batch_size, num_workers=num_workers,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold",        type=int, default=0)
    ap.add_argument("--epochs",      type=int, default=cfg.MAX_EPOCHS)
    ap.add_argument("--batch_size",  type=int, default=cfg.BATCH_SIZE)
    ap.add_argument("--all_folds",   action="store_true")
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()
    folds = range(cfg.NUM_FOLDS) if args.all_folds else [args.fold]
    for f in folds:
        run(fold_idx=f, num_epochs=args.epochs,
            batch_size=args.batch_size, num_workers=args.num_workers)
