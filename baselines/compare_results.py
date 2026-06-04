"""
Load all results.json from experiments/*/fold_*/results.json,
aggregate per model, and produce comparison / ablation tables + plots.

Outputs:
  outputs/tables/baseline_comparison.csv
  outputs/tables/ablation_study.csv
  outputs/tables/efficiency_table.csv
  outputs/figures/sensitivity_comparison.png
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

import src.config as cfg
from src.utils import save_json


# ---------------------------------------------------------------------------
# All model names (proposed + baselines + ablations)
# ---------------------------------------------------------------------------

ALL_MODELS = [
    "dr_litenet_effb0",
    "dr_litenet_mobv3",
    "effb0_baseline",
    "effb0_weighted",
    "mobv3_baseline",
    "resnet50_baseline",
    "dr_litenet_no_smote",
    "dr_litenet_weighted",
]

ABLATION_MODELS = [
    "dr_litenet_effb0",
    "dr_litenet_no_smote",
    "dr_litenet_weighted",
    "effb0_baseline",
]


# ---------------------------------------------------------------------------
# Load & aggregate results across folds
# ---------------------------------------------------------------------------

def load_model_results(model_name: str, num_folds: int = cfg.NUM_FOLDS) -> list[dict]:
    """Load all available fold results.json for a model. Returns list of dicts."""
    records = []
    for fold_idx in range(num_folds):
        rpath = cfg.experiment_path(model_name, fold_idx + 1) / cfg.RESULTS_JSON
        if rpath.exists():
            with open(rpath) as f:
                records.append(json.load(f))
    return records


def aggregate_model(records: list[dict]) -> dict:
    """
    Compute mean ± std for scalar metrics across folds.
    Returns a flat dict: {metric_mean: float, metric_std: float, ...}
    """
    if not records:
        return {}

    def _get(r, key, alt_key=None):
        v = r.get(key, r.get(alt_key, None) if alt_key else None)
        return float(v) if v is not None else None

    scalar_keys = [
        ("accuracy",  "val_accuracy",  "final_val_accuracy"),
        ("qwk",       "val_qwk",       "final_val_qwk"),
        ("macro_f1",  "val_macro_f1",  "final_val_macro_f1"),
        ("param_count", "param_count", None),
        ("inference_ms", "inference_time_ms", None),
    ]

    agg = {}
    for name, key, alt in scalar_keys:
        vals = [_get(r, key, alt) for r in records if _get(r, key, alt) is not None]
        if vals:
            agg[f"{name}_mean"] = round(float(np.mean(vals)), 6)
            agg[f"{name}_std"]  = round(float(np.std(vals)),  6)

    # Per-class sensitivity
    sens_lists = [r.get("val_per_class_sensitivity", []) for r in records if r.get("val_per_class_sensitivity")]
    if sens_lists:
        sens_arr = np.array(sens_lists)      # (n_folds, 5)
        for c in range(cfg.NUM_CLASSES):
            agg[f"sens_{c}_mean"] = round(float(sens_arr[:, c].mean()), 6)
            agg[f"sens_{c}_std"]  = round(float(sens_arr[:, c].std()),  6)

    # Per-class AUC
    auc_lists = [r.get("val_per_class_auc", []) for r in records if r.get("val_per_class_auc")]
    if auc_lists:
        auc_arr = np.array(auc_lists)
        for c in range(cfg.NUM_CLASSES):
            agg[f"auc_{c}_mean"] = round(float(auc_arr[:, c].mean()), 6)
            agg[f"auc_{c}_std"]  = round(float(auc_arr[:, c].std()),  6)

    agg["n_folds"] = len(records)
    return agg


# ---------------------------------------------------------------------------
# Table generators
# ---------------------------------------------------------------------------

def build_comparison_table(model_names: list[str]) -> pd.DataFrame:
    """
    Returns a DataFrame: model | accuracy | qwk | macro_f1 | sens_0..4 | auc_0..4 | params | ms
    Each metric shown as mean±std.
    """
    rows = []
    for mname in model_names:
        records = load_model_results(mname)
        if not records:
            continue
        agg = aggregate_model(records)
        row = {"model": mname}

        for metric in ("accuracy", "qwk", "macro_f1"):
            m = agg.get(f"{metric}_mean", None)
            s = agg.get(f"{metric}_std",  None)
            row[metric] = f"{m:.4f}±{s:.4f}" if m is not None else "—"

        for c in range(cfg.NUM_CLASSES):
            m = agg.get(f"sens_{c}_mean", None)
            s = agg.get(f"sens_{c}_std",  None)
            row[f"sens_{c}"] = f"{m:.3f}±{s:.3f}" if m is not None else "—"

        for c in range(cfg.NUM_CLASSES):
            m = agg.get(f"auc_{c}_mean", None)
            s = agg.get(f"auc_{c}_std",  None)
            row[f"auc_{c}"] = f"{m:.3f}±{s:.3f}" if m is not None else "—"

        pm = agg.get("param_count_mean", None)
        row["params"] = f"{int(pm):,}" if pm else "—"
        im = agg.get("inference_ms_mean", None)
        row["inference_ms"] = f"{im:.1f}" if im else "—"

        rows.append(row)
    return pd.DataFrame(rows)


def build_ablation_table(model_names: list[str]) -> pd.DataFrame:
    """Same structure but restricted to ablation models."""
    return build_comparison_table(model_names)


def build_efficiency_table(model_names: list[str]) -> pd.DataFrame:
    """model | param_count | inference_time_ms (mean across folds)"""
    rows = []
    for mname in model_names:
        records = load_model_results(mname)
        if not records:
            continue
        agg = aggregate_model(records)
        rows.append({
            "model":            mname,
            "param_count":      int(agg["param_count_mean"]) if "param_count_mean" in agg else "—",
            "inference_ms":     round(agg["inference_ms_mean"], 2) if "inference_ms_mean" in agg else "—",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plot: per-class sensitivity grouped bar chart
# ---------------------------------------------------------------------------

def plot_sensitivity_comparison(model_names: list[str]) -> Path:
    cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    model_sens = {}
    model_errs = {}
    for mname in model_names:
        records = load_model_results(mname)
        if not records:
            continue
        agg = aggregate_model(records)
        sens  = [agg.get(f"sens_{c}_mean", 0.0) for c in range(cfg.NUM_CLASSES)]
        errs  = [agg.get(f"sens_{c}_std",  0.0) for c in range(cfg.NUM_CLASSES)]
        if any(s > 0 for s in sens):
            model_sens[mname] = sens
            model_errs[mname] = errs

    if not model_sens:
        print("  [WARN] No sensitivity data found — skipping plot.")
        return None

    n_models  = len(model_sens)
    n_classes = cfg.NUM_CLASSES
    x         = np.arange(n_classes)
    width     = 0.8 / max(n_models, 1)
    colors    = plt.cm.tab10(np.linspace(0, 0.7, n_models))
    class_labels = [cfg.CLASS_NAMES[c] for c in range(n_classes)]

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (mname, sens) in enumerate(model_sens.items()):
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, sens, width=width * 0.92,
               yerr=model_errs[mname], capsize=3,
               label=mname, color=colors[i], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(class_labels, rotation=12, ha="right")
    ax.set_ylabel("Sensitivity (Recall)")
    ax.set_ylim(0, 1.12)
    ax.set_title("Per-Class Sensitivity Comparison (mean ± std across folds)")
    ax.legend(fontsize=7, ncol=2)
    ax.axhline(0.8, color="gray", linestyle="--", linewidth=0.7, label="_nolegend_")
    plt.tight_layout()

    out_path = cfg.FIGURES_DIR / "sensitivity_comparison.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame, title: str = "Comparison"):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    cols = ["model", "accuracy", "qwk", "macro_f1"]
    sens_cols = [c for c in df.columns if c.startswith("sens_")]
    print(df[cols + sens_cols].to_string(index=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_comparison(model_names: list[str] = None) -> dict:
    if model_names is None:
        model_names = ALL_MODELS

    cfg.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Baseline comparison
    df_cmp = build_comparison_table(model_names)
    if not df_cmp.empty:
        out = cfg.TABLES_DIR / "baseline_comparison.csv"
        df_cmp.to_csv(out, index=False)
        print(f"  Saved: {out}")
        print_summary(df_cmp, "Baseline Comparison")

    # Ablation study
    df_abl = build_ablation_table(ABLATION_MODELS)
    if not df_abl.empty:
        out = cfg.TABLES_DIR / "ablation_study.csv"
        df_abl.to_csv(out, index=False)
        print(f"  Saved: {out}")
        print_summary(df_abl, "Ablation Study")

    # Efficiency table
    df_eff = build_efficiency_table(model_names)
    if not df_eff.empty:
        out = cfg.TABLES_DIR / "efficiency_table.csv"
        df_eff.to_csv(out, index=False)
        print(f"  Saved: {out}")
        print(f"\n--- Efficiency ---\n{df_eff.to_string(index=False)}")

    # Sensitivity plot
    sens_path = plot_sensitivity_comparison(model_names)
    if sens_path:
        print(f"  Saved: {sens_path}")

    return {
        "comparison": df_cmp.to_dict(orient="records") if not df_cmp.empty else [],
        "ablation":   df_abl.to_dict(orient="records") if not df_abl.empty else [],
        "efficiency": df_eff.to_dict(orient="records") if not df_eff.empty else [],
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None,
                    help="Subset of model names to compare (default: all)")
    args = ap.parse_args()
    run_comparison(model_names=args.models)
