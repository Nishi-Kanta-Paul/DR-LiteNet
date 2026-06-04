"""
DR-LiteNet — unified CLI entry point.

Modes:
  train    — run 5-fold training (Phase 1 + Phase 2) for a given model.
  evaluate — load best checkpoints, compute all metrics, generate plots/tables.
  infer    — run inference on a single image or directory.
  explain  — generate Grad-CAM visualisations from best checkpoint.
  compare  — load all experiment results and produce comparison/ablation tables.

Examples
--------
  # Train DR-LiteNet (EfficientNetB0), all folds:
  python src/main.py --mode train --model dr_litenet_effb0

  # Train a single fold quickly (sanity / debug):
  python src/main.py --mode train --model dr_litenet_effb0 --fold 0 \\
                     --epochs 2 --batch_size 4

  # Evaluate after training:
  python src/main.py --mode evaluate --model dr_litenet_effb0

  # Run inference on a directory of images:
  python src/main.py --mode infer --model dr_litenet_effb0 \\
                     --input data/val_images/val_images/ --gradcam

  # Generate Grad-CAM figures:
  python src/main.py --mode explain --model dr_litenet_effb0 --fold 0

  # Build comparison/ablation tables from all trained models:
  python src/main.py --mode compare
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.config as cfg
from src.utils import get_device, set_seed


# ---------------------------------------------------------------------------
# Mode implementations
# ---------------------------------------------------------------------------

def mode_train(args):
    """Train one model for all folds (or a single fold)."""
    from src.train import train_single_fold, train_all_folds

    # Resolve backbone from model name
    backbone = _backbone_from_model(args.model)

    common_kwargs = dict(
        model_name=args.model,
        backbone=backbone,
        phase1_epochs=args.p1_epochs,
        phase2_epochs=args.p2_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=cfg.SEED,
    )

    if args.fold is not None:
        print(f"[main] Training fold {args.fold} for model '{args.model}'")
        train_single_fold(fold_idx=args.fold, **common_kwargs)
    else:
        print(f"[main] Training all {cfg.NUM_FOLDS} folds for model '{args.model}'")
        train_all_folds(**common_kwargs)


def mode_evaluate(args):
    """Load best checkpoints and run full evaluation + plot generation."""
    from src.evaluate import run_full_evaluation

    device   = get_device()
    backbone = _backbone_from_model(args.model)
    folds    = [args.fold] if args.fold is not None else None   # None → all available

    print(f"[main] Evaluating model '{args.model}' on device '{device}'")
    run_full_evaluation(
        model_name=args.model,
        backbone=backbone,
        num_folds=cfg.NUM_FOLDS if folds is None else 1,
        batch_size=args.batch_size,
        device=device,
        num_workers=args.num_workers,
    )


def mode_infer(args):
    """Run inference on images or IDRiD external validation."""
    from src.inference import load_model, run_inference, evaluate_on_idrid

    device    = get_device()
    ckpt_path = _resolve_checkpoint(args.model, args.fold or 1)

    model = load_model(ckpt_path, backbone=_backbone_from_model(args.model), device=device)

    if args.idrid:
        evaluate_on_idrid(model, device, batch_size=args.batch_size,
                          num_workers=args.num_workers)
    elif args.input:
        run_inference(
            model=model,
            input_path=Path(args.input),
            device=device,
            generate_gradcam=args.gradcam,
            save_dir=Path(args.output) if args.output else cfg.PREDICTIONS_DIR,
        )
    else:
        print("[main] Provide --input <path> or --idrid flag.")
        sys.exit(1)


def mode_explain(args):
    """Generate Grad-CAM visualisations from the best checkpoint."""
    from src.dataset import get_dataloaders
    from src.explainability import (
        generate_gradcam_for_dataset,
        generate_gradcam_misclassified,
    )
    from src.model import build_model
    from src.utils import load_checkpoint

    device    = get_device()
    fold_idx  = args.fold if args.fold is not None else 0
    ckpt_path = _resolve_checkpoint(args.model, fold_idx + 1)
    backbone  = _backbone_from_model(args.model)

    print(f"[main] Generating Grad-CAM for '{args.model}' fold {fold_idx+1}")
    model = build_model(backbone_name=backbone, pretrained=False, device=device)
    load_checkpoint(ckpt_path, model, device)

    _, val_loader = get_dataloaders(
        fold_idx=fold_idx, batch_size=args.batch_size,
        use_handcrafted=True, num_workers=args.num_workers,
    )
    generate_gradcam_for_dataset(
        model=model, val_loader=val_loader, device=device,
        num_samples_per_class=3,
        save_dir=cfg.GRAD_CAM_DIR,
        model_name=args.model,
    )
    generate_gradcam_misclassified(
        model=model, val_loader=val_loader, device=device,
        num_samples=10, save_dir=cfg.GRAD_CAM_DIR / "misclassified",
    )


def mode_compare(args):
    """Load all results.json and produce comparison / ablation tables."""
    from baselines.compare_results import run_comparison
    print("[main] Building comparison tables...")
    run_comparison()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _backbone_from_model(model_name: str) -> str:
    """Infer backbone name from experiment model name."""
    if "mobv3" in model_name:
        return "mobilenet_v3_small"
    if "resnet50" in model_name:
        return "resnet50"
    return "efficientnet_b0"   # default


def _resolve_checkpoint(model_name: str, fold: int) -> Path:
    """Return best_model.pth path; raise if missing."""
    p = cfg.checkpoint_path(model_name, fold)
    if not p.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {p}\n"
            f"Run: python src/main.py --mode train --model {model_name}"
        )
    return p


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python src/main.py",
        description="DR-LiteNet — unified CLI for training, evaluation, inference, and explainability.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--mode", required=True,
        choices=["train", "evaluate", "infer", "explain", "compare"],
        help="Pipeline mode.",
    )
    p.add_argument(
        "--model", default="dr_litenet_effb0",
        choices=cfg.MODEL_NAMES,
        help="Experiment model name (default: dr_litenet_effb0).",
    )
    p.add_argument(
        "--fold", type=int, default=None,
        help="Specific fold index 0–4. Omit to run all folds.",
    )
    # Training
    p.add_argument("--p1_epochs",   type=int, default=cfg.PHASE1_EPOCHS,
                   help="Phase 1 epochs (frozen backbone + SMOTE).")
    p.add_argument("--p2_epochs",   type=int, default=cfg.PHASE2_EPOCHS,
                   help="Phase 2 epochs (end-to-end fine-tuning).")
    p.add_argument("--batch_size",  type=int, default=cfg.BATCH_SIZE)
    p.add_argument("--num_workers", type=int, default=2)
    # Inference
    p.add_argument("--input",   type=str, default=None,
                   help="(infer) Image file or directory.")
    p.add_argument("--output",  type=str, default=None,
                   help="(infer) Output directory for predictions.")
    p.add_argument("--gradcam", action="store_true",
                   help="(infer) Generate Grad-CAM alongside predictions.")
    p.add_argument("--idrid",   action="store_true",
                   help="(infer) Run IDRiD external validation.")
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = build_parser()
    args   = parser.parse_args()

    set_seed(cfg.SEED)

    dispatch = {
        "train":    mode_train,
        "evaluate": mode_evaluate,
        "infer":    mode_infer,
        "explain":  mode_explain,
        "compare":  mode_compare,
    }
    dispatch[args.mode](args)


if __name__ == "__main__":
    main()
