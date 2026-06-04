"""
Evaluation & metrics for DR-LiteNet.

Functions:
  compute_all_metrics(y_true, y_pred, y_probs)  → full metric dict
  evaluate_fold(model, val_loader, device, ...)  → inference + save results.json
  aggregate_results(model_name, num_folds)       → mean±std tables

Plot generators (all saved to outputs/figures/):
  plot_confusion_matrix(cm, fold_idx, model_name, averaged)
  plot_roc_curves(y_true, y_probs, fold_idx, model_name)
  plot_training_curves(log_csv, fold_idx, model_name)
  plot_class_distribution(counts_before, counts_after)
  plot_sensitivity_bar(sensitivities_dict)       ← placeholder for Stage 7
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from tqdm import tqdm

import src.config as cfg
from src.utils import save_json


# ---------------------------------------------------------------------------
# 1. Metric computation
# ---------------------------------------------------------------------------

def compute_all_metrics(
    y_true: list,
    y_pred: list,
    y_probs: np.ndarray,          # (N, num_classes) softmax probabilities
) -> dict:
    """
    Compute the full evaluation metric suite.

    Returns a dict with:
      accuracy, qwk, macro_f1,
      per_class_sensitivity [list length 5],
      per_class_precision   [list length 5],
      per_class_auc         [list length 5],
      confusion_matrix      [5×5 list of lists],
      support               [list length 5]  — true counts per class
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    classes = list(range(cfg.NUM_CLASSES))

    # Scalar metrics
    accuracy = float(accuracy_score(y_true, y_pred))
    qwk      = float(cohen_kappa_score(y_true, y_pred, weights="quadratic")) \
                if len(np.unique(y_true)) > 1 else 0.0
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # Per-class
    per_class_sens = recall_score(
        y_true, y_pred, labels=classes, average=None, zero_division=0
    ).tolist()
    per_class_prec = precision_score(
        y_true, y_pred, labels=classes, average=None, zero_division=0
    ).tolist()

    # AUC-ROC (one-vs-rest)
    y_bin = label_binarize(y_true, classes=classes)  # (N, 5)
    per_class_auc = []
    for c in classes:
        if y_bin[:, c].sum() == 0:
            per_class_auc.append(0.0)
        else:
            auc = roc_auc_score(y_bin[:, c], y_probs[:, c])
            per_class_auc.append(float(auc))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    # Support
    support = [int((y_true == c).sum()) for c in classes]

    return {
        "accuracy":              accuracy,
        "qwk":                   qwk,
        "macro_f1":              macro_f1,
        "per_class_sensitivity": per_class_sens,
        "per_class_precision":   per_class_prec,
        "per_class_auc":         per_class_auc,
        "confusion_matrix":      cm.tolist(),
        "support":               support,
    }


# ---------------------------------------------------------------------------
# 2. Single-fold evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_fold(
    model,
    val_loader: DataLoader,
    device: torch.device,
    fold_idx: int,
    model_name: str = "dr_litenet_effb0",
    save_results: bool = True,
) -> dict:
    """
    Run inference on val_loader, compute all metrics, optionally save results.json.
    Returns the full metrics dict.
    """
    model.eval()

    all_preds, all_labels, all_probs = [], [], []

    for imgs, hc, labels in tqdm(val_loader, desc=f"  Evaluating fold {fold_idx+1}", leave=False):
        imgs   = imgs.to(device)
        hc     = hc.to(device)
        labels = labels.to(device)

        logits = model(imgs, hc)
        probs  = F.softmax(logits, dim=1)
        preds  = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        all_probs.append(probs.cpu().numpy())

    y_probs = np.concatenate(all_probs, axis=0)   # (N, 5)
    metrics = compute_all_metrics(all_labels, all_preds, y_probs)

    if save_results:
        exp_dir = cfg.experiment_path(model_name, fold_idx + 1)
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Build results.json matching STABLE §13 structure
        results = {
            "model_name":               model_name,
            "fold":                     fold_idx + 1,
            "val_accuracy":             round(metrics["accuracy"], 6),
            "val_qwk":                  round(metrics["qwk"], 6),
            "val_macro_f1":             round(metrics["macro_f1"], 6),
            "val_per_class_sensitivity": [round(v, 6) for v in metrics["per_class_sensitivity"]],
            "val_per_class_precision":   [round(v, 6) for v in metrics["per_class_precision"]],
            "val_per_class_auc":         [round(v, 6) for v in metrics["per_class_auc"]],
            "confusion_matrix":          metrics["confusion_matrix"],
            "support":                   metrics["support"],
        }
        save_json(results, exp_dir / cfg.RESULTS_JSON)

    return {**metrics, "y_true": all_labels, "y_pred": all_preds, "y_probs": y_probs}


# ---------------------------------------------------------------------------
# 3. Cross-fold aggregation
# ---------------------------------------------------------------------------

def aggregate_results(
    model_name: str = "dr_litenet_effb0",
    num_folds: int = cfg.NUM_FOLDS,
) -> dict:
    """
    Load results.json for each fold, compute mean ± std across folds.
    Save averaged_results.csv and per_fold_results.csv to outputs/tables/.
    Returns the aggregation dict.
    """
    cfg.TABLES_DIR.mkdir(parents=True, exist_ok=True)

    fold_records = []
    for fold_idx in range(num_folds):
        rpath = cfg.experiment_path(model_name, fold_idx + 1) / cfg.RESULTS_JSON
        if not rpath.exists():
            print(f"  [WARN] Missing results.json for fold {fold_idx+1}: {rpath}")
            continue
        with open(rpath) as f:
            r = json.load(f)
        row = {
            "fold":      r.get("fold", fold_idx + 1),
            "accuracy":  r.get("val_accuracy", r.get("final_val_accuracy", 0.0)),
            "qwk":       r.get("val_qwk",      r.get("final_val_qwk", 0.0)),
            "macro_f1":  r.get("val_macro_f1", r.get("final_val_macro_f1", 0.0)),
        }
        # Per-class sensitivity
        sens = r.get("val_per_class_sensitivity", [0.0] * cfg.NUM_CLASSES)
        for c in range(cfg.NUM_CLASSES):
            row[f"sens_{c}"] = sens[c] if c < len(sens) else 0.0
        # Per-class AUC
        auc = r.get("val_per_class_auc", [0.0] * cfg.NUM_CLASSES)
        for c in range(cfg.NUM_CLASSES):
            row[f"auc_{c}"] = auc[c] if c < len(auc) else 0.0
        fold_records.append(row)

    if not fold_records:
        print("  [WARN] No fold results found — skipping aggregation.")
        return {}

    df = pd.DataFrame(fold_records)

    # Per-fold table
    per_fold_path = cfg.TABLES_DIR / "per_fold_results.csv"
    df.to_csv(per_fold_path, index=False)
    print(f"  Saved: {per_fold_path}")

    # Averaged table
    numeric_cols = [c for c in df.columns if c != "fold"]
    avg_rows = []
    for col in numeric_cols:
        avg_rows.append({
            "metric": col,
            "mean":   round(float(df[col].mean()), 6),
            "std":    round(float(df[col].std()), 6),
        })
    df_avg = pd.DataFrame(avg_rows)
    avg_path = cfg.TABLES_DIR / "averaged_results.csv"
    df_avg.to_csv(avg_path, index=False)
    print(f"  Saved: {avg_path}")

    return {
        "per_fold":  df.to_dict(orient="records"),
        "averaged":  df_avg.to_dict(orient="records"),
    }


# ---------------------------------------------------------------------------
# 4a. Confusion matrix heatmap
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    fold_idx: int = None,
    model_name: str = "dr_litenet_effb0",
    averaged: bool = False,
    normalise: bool = True,
) -> Path:
    """
    Save a seaborn confusion-matrix heatmap.
    If averaged=True, saves as confusion_matrix_avg.png, otherwise fold_<k>.png.
    Returns the saved path.
    """
    cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if normalise:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_plot = cm.astype(float) / row_sums
        fmt, vmin, vmax = ".2f", 0.0, 1.0
        title_suffix = " (row-normalised)"
    else:
        cm_plot = cm.astype(int)
        fmt, vmin, vmax = "d", 0, int(cm.max())
        title_suffix = " (counts)"

    labels = [cfg.CLASS_NAMES[i] for i in range(cfg.NUM_CLASSES)]

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm_plot, annot=True, fmt=fmt, cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        vmin=vmin, vmax=vmax, ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    tag = "avg" if averaged else f"fold_{fold_idx+1}"
    ax.set_title(f"Confusion Matrix — {model_name} ({tag}){title_suffix}", fontsize=11)
    plt.tight_layout()

    fname = "confusion_matrix_avg.png" if averaged else f"confusion_matrix_fold_{fold_idx+1}.png"
    out_path = cfg.FIGURES_DIR / fname
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 4b. ROC curves
# ---------------------------------------------------------------------------

def plot_roc_curves(
    y_true: list,
    y_probs: np.ndarray,
    fold_idx: int = 0,
    model_name: str = "dr_litenet_effb0",
) -> Path:
    """Per-class one-vs-rest ROC curves. Saved as roc_curves_fold_<k>.png."""
    cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(y_true, dtype=int)
    y_bin  = label_binarize(y_true, classes=list(range(cfg.NUM_CLASSES)))

    colors = plt.cm.tab10(np.linspace(0, 0.5, cfg.NUM_CLASSES))
    fig, ax = plt.subplots(figsize=(8, 6))

    for c in range(cfg.NUM_CLASSES):
        if y_bin[:, c].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, c], y_probs[:, c])
        auc_val = roc_auc_score(y_bin[:, c], y_probs[:, c])
        ax.plot(fpr, tpr, color=colors[c],
                label=f"{cfg.CLASS_NAMES[c]} (AUC={auc_val:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curves — {model_name} fold_{fold_idx+1}", fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()

    fname = f"roc_curves_fold_{fold_idx+1}.png"
    out_path = cfg.FIGURES_DIR / fname
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 4c. Training curves (loss + QWK) — read from train_log.csv
# ---------------------------------------------------------------------------

def plot_training_curves(
    fold_idx: int,
    model_name: str = "dr_litenet_effb0",
) -> tuple[Path, Path]:
    """
    Plot loss curves and QWK curves from train_log.csv.
    Returns (loss_curve_path, qwk_curve_path).
    """
    cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    log_csv = cfg.log_path(model_name, fold_idx + 1) / cfg.TRAIN_LOG_CSV
    if not log_csv.exists():
        print(f"  [WARN] train_log.csv not found: {log_csv}")
        return None, None

    df = pd.read_csv(log_csv)
    epochs = df["epoch"].tolist()

    # --- Loss curves ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, df["train_loss"], label="Train Loss", marker="o", markersize=3)
    ax.plot(epochs, df["val_loss"],   label="Val Loss",   marker="s", markersize=3)
    # Shade Phase 1 / Phase 2 regions
    if "phase" in df.columns:
        p1_end = df[df["phase"] == 1]["epoch"].max()
        p2_rows = df[df["phase"] == 2]
        if not p2_rows.empty:
            ax.axvline(p1_end + 0.5, color="gray", linestyle="--",
                       linewidth=0.8, label="Phase 1→2")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Loss Curves — {model_name} fold_{fold_idx+1}")
    ax.legend()
    plt.tight_layout()
    loss_path = cfg.FIGURES_DIR / f"loss_curves_fold_{fold_idx+1}.png"
    fig.savefig(loss_path, dpi=120)
    plt.close(fig)

    # --- QWK curves ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, df["val_qwk"],      label="Val QWK",    marker="s", markersize=3)
    ax.plot(epochs, df["val_macro_f1"], label="Val Macro F1", marker="^", markersize=3)
    if "phase" in df.columns and not p2_rows.empty:
        ax.axvline(p1_end + 0.5, color="gray", linestyle="--",
                   linewidth=0.8, label="Phase 1→2")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title(f"QWK & F1 Curves — {model_name} fold_{fold_idx+1}")
    ax.legend()
    plt.tight_layout()
    qwk_path = cfg.FIGURES_DIR / f"qwk_curves_fold_{fold_idx+1}.png"
    fig.savefig(qwk_path, dpi=120)
    plt.close(fig)

    return loss_path, qwk_path


# ---------------------------------------------------------------------------
# 4d. Class distribution before/after SMOTE
# ---------------------------------------------------------------------------

def plot_class_distribution(
    counts_before: dict,
    counts_after:  dict,
    save_path: Path = None,
) -> Path:
    """Bar chart: class counts before and after SMOTE."""
    cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if save_path is None:
        save_path = cfg.FIGURES_DIR / "class_distribution_smote.png"

    classes = list(range(cfg.NUM_CLASSES))
    labels  = [cfg.CLASS_NAMES[c] for c in classes]
    before  = [counts_before.get(c, 0) for c in classes]
    after   = [counts_after.get(c,  0) for c in classes]

    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width/2, before, width, label="Before SMOTE", color="steelblue")
    ax.bar(x + width/2, after,  width, label="After SMOTE",  color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Sample count")
    ax.set_title("Class Distribution Before / After SMOTE")
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------
# 4e. Per-class sensitivity bar chart (placeholder — updated in Stage 7)
# ---------------------------------------------------------------------------

def plot_sensitivity_bar(
    sensitivities_dict: dict,
    save_path: Path = None,
) -> Path:
    """
    Bar chart comparing per-class sensitivity across models.
    sensitivities_dict: {model_name: [sens_0, sens_1, ..., sens_4]}
    """
    cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if save_path is None:
        save_path = cfg.FIGURES_DIR / "sensitivity_comparison.png"

    classes = [cfg.CLASS_NAMES[c] for c in range(cfg.NUM_CLASSES)]
    x = np.arange(len(classes))
    n_models = len(sensitivities_dict)
    width = 0.8 / max(n_models, 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.tab10(np.linspace(0, 0.6, n_models))
    for i, (mname, sens) in enumerate(sensitivities_dict.items()):
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, sens, width=width * 0.9, label=mname, color=colors[i])

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=15, ha="right")
    ax.set_ylabel("Sensitivity (Recall)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-Class Sensitivity Comparison")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------
# 5. Full evaluation pipeline for one model (all folds)
# ---------------------------------------------------------------------------

def run_full_evaluation(
    model_name: str = "dr_litenet_effb0",
    backbone:   str = cfg.BACKBONE,
    num_folds:  int = cfg.NUM_FOLDS,
    batch_size: int = cfg.BATCH_SIZE,
    device:     torch.device = None,
    num_workers: int = 2,
) -> dict:
    """
    For each fold: load best checkpoint → evaluate → generate per-fold plots.
    Then aggregate across folds and generate summary plots/tables.
    """
    from src.dataset import get_dataloaders
    from src.model import build_model
    from src.utils import load_checkpoint, get_device

    if device is None:
        device = get_device()

    all_cms   = []
    all_sens  = []
    fold_results = []

    for fold_idx in range(num_folds):
        ckpt_path = cfg.checkpoint_path(model_name, fold_idx + 1)
        if not ckpt_path.exists():
            print(f"  [SKIP] No checkpoint for fold {fold_idx+1}: {ckpt_path}")
            continue

        print(f"\n[Fold {fold_idx+1}]")
        _, val_loader = get_dataloaders(
            fold_idx=fold_idx, batch_size=batch_size,
            use_handcrafted=True, num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )

        model = build_model(backbone_name=backbone, pretrained=False, device=device)
        load_checkpoint(ckpt_path, model, device)

        result = evaluate_fold(
            model=model, val_loader=val_loader, device=device,
            fold_idx=fold_idx, model_name=model_name, save_results=True,
        )
        fold_results.append(result)

        # Per-fold plots
        plot_confusion_matrix(
            np.array(result["confusion_matrix"]),
            fold_idx=fold_idx, model_name=model_name, averaged=False,
        )
        plot_roc_curves(result["y_true"], result["y_probs"], fold_idx, model_name)
        plot_training_curves(fold_idx, model_name)

        all_cms.append(np.array(result["confusion_matrix"]))
        all_sens.append(result["per_class_sensitivity"])

    # Averaged confusion matrix
    if all_cms:
        avg_cm = np.mean(all_cms, axis=0)
        plot_confusion_matrix(avg_cm, averaged=True, model_name=model_name)

    # Sensitivity comparison (single model, for now)
    if all_sens:
        mean_sens = np.mean(all_sens, axis=0).tolist()
        plot_sensitivity_bar({model_name: mean_sens})

    # SMOTE class distribution (from APTOS training set)
    import pandas as pd2
    from collections import Counter
    df_tr = pd.read_csv(cfg.APTOS_TRAIN_CSV)
    before = dict(Counter(df_tr["diagnosis"].tolist()))
    # Approximate "after": majority count replicated for all classes
    maj = max(before.values())
    after = {c: maj for c in range(cfg.NUM_CLASSES)}
    plot_class_distribution(before, after)

    # Aggregate tables
    agg = aggregate_results(model_name=model_name, num_folds=num_folds)

    return {"fold_results": fold_results, "aggregation": agg}


# ---------------------------------------------------------------------------
# Sanity check — run as script
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import torch
    from src.dataset import get_dataloaders
    from src.model import build_model
    from src.utils import get_device, load_checkpoint

    print("=" * 60)
    print("evaluate.py sanity check — fold 1")
    print("=" * 60)

    device    = get_device()
    fold_idx  = 0
    model_name = "dr_litenet_effb0"
    backbone   = "efficientnet_b0"

    # Build model + load checkpoint
    ckpt_path = cfg.checkpoint_path(model_name, fold_idx + 1)
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

    model = build_model(backbone_name=backbone, pretrained=False, device=device)
    ckpt  = load_checkpoint(ckpt_path, model, device)
    print(f"  Loaded checkpoint from epoch {ckpt.get('epoch', '?')}, "
          f"val_qwk={ckpt.get('val_qwk', '?')}")

    # DataLoader (batch_size=8)
    _, val_loader = get_dataloaders(
        fold_idx=fold_idx, batch_size=8, use_handcrafted=True, num_workers=0
    )

    # Evaluate
    result = evaluate_fold(
        model=model, val_loader=val_loader, device=device,
        fold_idx=fold_idx, model_name=model_name, save_results=True,
    )

    print("\n--- Metric summary (fold 1) ---")
    print(f"  Accuracy  : {result['accuracy']:.4f}")
    print(f"  QWK       : {result['qwk']:.4f}")
    print(f"  Macro F1  : {result['macro_f1']:.4f}")
    for c in range(cfg.NUM_CLASSES):
        print(f"  {cfg.CLASS_NAMES[c]:20s}: "
              f"sens={result['per_class_sensitivity'][c]:.3f}  "
              f"prec={result['per_class_precision'][c]:.3f}  "
              f"auc={result['per_class_auc'][c]:.3f}")

    # Verify results.json structure
    import json as _json
    rpath = cfg.experiment_path(model_name, fold_idx + 1) / cfg.RESULTS_JSON
    with open(rpath) as f:
        saved = _json.load(f)
    required_keys = [
        "model_name", "fold", "val_accuracy", "val_qwk", "val_macro_f1",
        "val_per_class_sensitivity", "val_per_class_precision", "val_per_class_auc",
        "confusion_matrix", "support",
    ]
    for k in required_keys:
        assert k in saved, f"Missing key in results.json: {k}"
    print(f"\n  [OK] results.json keys: {list(saved.keys())}")

    # Plots
    print("\n--- Generating plots ---")
    cm = np.array(result["confusion_matrix"])

    p1 = plot_confusion_matrix(cm, fold_idx=0, model_name=model_name, averaged=False)
    print(f"  [OK] {p1}")

    p2 = plot_confusion_matrix(cm, averaged=True, model_name=model_name)
    print(f"  [OK] {p2}")

    p3 = plot_roc_curves(result["y_true"], result["y_probs"], fold_idx=0,
                         model_name=model_name)
    print(f"  [OK] {p3}")

    p4, p5 = plot_training_curves(fold_idx=0, model_name=model_name)
    if p4:
        print(f"  [OK] {p4}")
        print(f"  [OK] {p5}")

    import pandas as pd2
    from collections import Counter
    df_tr = pd.read_csv(cfg.APTOS_TRAIN_CSV)
    before = dict(Counter(df_tr["diagnosis"].tolist()))
    maj    = max(before.values())
    after  = {c: maj for c in range(cfg.NUM_CLASSES)}
    p6 = plot_class_distribution(before, after)
    print(f"  [OK] {p6}")

    p7 = plot_sensitivity_bar({model_name: result["per_class_sensitivity"]})
    print(f"  [OK] {p7}")

    # Aggregation tables (single fold available)
    print("\n--- Aggregation tables ---")
    agg = aggregate_results(model_name=model_name, num_folds=1)

    # List all generated files
    print("\n--- Generated files ---")
    for f in sorted(cfg.FIGURES_DIR.glob("*.png")):
        print(f"  {f}")
    for f in sorted(cfg.TABLES_DIR.glob("*.csv")):
        print(f"  {f}")

    print("\n[ALL SANITY CHECKS PASSED]")
