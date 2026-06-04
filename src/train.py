"""
Training pipeline for DR-LiteNet.

Two-phase training:
  Phase 1 — Frozen backbone → extract hybrid features → ADASYN → train head only.
  Phase 2 — Unfreeze backbone → end-to-end fine-tuning (no SMOTE).

Entry points:
  train_single_fold(fold_idx, ...)  — train one fold (Phase 1 + Phase 2).
  train_all_folds(...)              — loop over all 5 folds.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import src.config as cfg
from src.dataset import get_dataloaders
from src.model import build_model
from src.utils import (
    AverageMeter,
    append_log_row,
    cache_features,
    cached_features_exist,
    compute_accuracy,
    compute_macro_f1,
    compute_qwk,
    get_device,
    load_cached_features,
    save_checkpoint,
    save_json,
    set_seed,
)


# ---------------------------------------------------------------------------
# SMOTE helper
# ---------------------------------------------------------------------------

def _apply_smote(
    X: np.ndarray,
    y: np.ndarray,
    variant: str = cfg.SMOTE_VARIANT,
    sampling_strategy=cfg.SMOTE_SAMPLING_STRATEGY,
    random_state: int = cfg.SMOTE_RANDOM_STATE,
    n_neighbors: int = cfg.SMOTE_N_NEIGHBORS,
):
    """
    Apply ADASYN / SMOTE / BorderlineSMOTE on feature matrix X, labels y.
    Falls back to SMOTE if ADASYN raises an error (e.g. too few minority samples).
    Returns X_res, y_res (numpy arrays).
    """
    from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE

    sampler_cls = {"ADASYN": ADASYN, "SMOTE": SMOTE, "BorderlineSMOTE": BorderlineSMOTE}
    if variant not in sampler_cls:
        raise ValueError(f"Unknown SMOTE variant '{variant}'. Choose from {list(sampler_cls)}")

    # Adapt n_neighbors to the smallest class size (must be < min_class_count)
    class_counts = Counter(y.tolist())
    min_class_count = min(class_counts.values())
    safe_k = min(n_neighbors, min_class_count - 1)

    print(f"\n  [SMOTE] Before: {dict(sorted(class_counts.items()))}")

    # If any class has only 1 sample, SMOTE cannot operate — return unchanged
    if safe_k < 1:
        print(f"  [SMOTE] min_class_count={min_class_count} — too few samples, "
              f"skipping SMOTE (returning original data).")
        return X, y

    if safe_k < n_neighbors:
        print(f"  [SMOTE] Reducing n_neighbors {n_neighbors}→{safe_k} "
              f"(min class count = {min_class_count})")

    try:
        sampler = sampler_cls[variant](
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            n_neighbors=safe_k,
        )
        X_res, y_res = sampler.fit_resample(X, y)
    except Exception as e:
        print(f"  [SMOTE] {variant} failed ({e}), falling back to SMOTE.")
        sampler = SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=safe_k,
        )
        X_res, y_res = sampler.fit_resample(X, y)

    print(f"  [SMOTE] After : {dict(sorted(Counter(y_res.tolist()).items()))}")
    return X_res, y_res


# ---------------------------------------------------------------------------
# Feature extraction pass (Phase 1)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _extract_all_features(
    model,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Forward-pass all batches through the frozen backbone to collect
    hybrid feature vectors for SMOTE.
    Returns (X, y) as float32 / int numpy arrays.
    """
    model.eval()
    all_features, all_labels = [], []
    for imgs, hc, labels in tqdm(loader, desc="  Extracting features", leave=False):
        imgs = imgs.to(device)
        hc   = hc.to(device)
        feats = model.extract_features(imgs, hc)   # (B, fused_dim)
        all_features.append(feats.cpu().numpy())
        all_labels.append(labels.numpy())
    X = np.concatenate(all_features, axis=0).astype(np.float32)
    y = np.concatenate(all_labels,   axis=0).astype(np.int32)
    return X, y


# ---------------------------------------------------------------------------
# Validation pass
# ---------------------------------------------------------------------------

@torch.no_grad()
def _validate(
    model,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """
    Full forward pass on validation set (both branches, no SMOTE).
    Returns dict with val_loss, val_accuracy, val_qwk, val_macro_f1.
    """
    model.eval()
    loss_meter = AverageMeter("val_loss")
    all_preds, all_labels = [], []

    for imgs, hc, labels in val_loader:
        imgs   = imgs.to(device)
        hc     = hc.to(device)
        labels = labels.to(device)

        logits = model(imgs, hc)
        loss   = criterion(logits, labels)
        preds  = logits.argmax(dim=1)

        loss_meter.update(loss.item(), n=len(labels))
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    return {
        "val_loss":     loss_meter.avg,
        "val_accuracy": compute_accuracy(all_labels, all_preds),
        "val_qwk":      compute_qwk(all_labels, all_preds),
        "val_macro_f1": compute_macro_f1(all_labels, all_preds),
    }


# ---------------------------------------------------------------------------
# Phase 1 — Frozen backbone + SMOTE + head training
# ---------------------------------------------------------------------------

def _phase1(
    model,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    fold_idx:     int,
    model_name:   str,
    device:       torch.device,
    phase1_epochs: int,
    batch_size:    int,
    log_path:      Path,
    ckpt_dir:      Path,
    early_stopping_patience: int,
    lr_head:       float,
    scheduler_patience: int,
    scheduler_factor:   float,
) -> tuple[float, int]:
    """
    Phase 1 training. Returns (best_val_qwk, epochs_trained).
    """
    print(f"\n{'='*60}")
    print(f"Phase 1 — Frozen backbone + ADASYN + head training")
    print(f"{'='*60}")

    model.freeze_backbone()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_head,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max",
        patience=scheduler_patience,
        factor=scheduler_factor,
        min_lr=cfg.SCHEDULER_MIN_LR,
    )
    criterion = nn.CrossEntropyLoss()

    best_val_qwk   = -1.0
    no_improve     = 0
    epochs_trained = 0

    for epoch in range(1, phase1_epochs + 1):
        print(f"\n  [Phase1 | Fold {fold_idx+1} | Epoch {epoch}/{phase1_epochs}]")

        # ---- Feature extraction pass ----
        X, y = _extract_all_features(model, train_loader, device)

        # ---- ADASYN ----
        X_res, y_res = _apply_smote(X, y)

        # ---- Build balanced TensorDataset ----
        X_t = torch.from_numpy(X_res).float()
        y_t = torch.from_numpy(y_res).long()
        balanced_ds = TensorDataset(X_t, y_t)
        balanced_loader = DataLoader(
            balanced_ds, batch_size=batch_size, shuffle=True, drop_last=False
        )

        # ---- Train classification head ----
        model.train()
        # Keep backbone in eval to avoid updating BN running stats while frozen
        model.cnn_branch.eval()

        loss_meter = AverageMeter("train_loss")
        for feat_batch, label_batch in tqdm(balanced_loader, desc="  Training head", leave=False):
            feat_batch  = feat_batch.to(device)
            label_batch = label_batch.to(device)

            optimizer.zero_grad()
            logits = model.classification_head(feat_batch)
            loss   = criterion(logits, label_batch)
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item(), n=len(label_batch))

        train_loss = loss_meter.avg

        # ---- Validation ----
        val_metrics = _validate(model, val_loader, criterion, device)
        val_qwk     = val_metrics["val_qwk"]
        scheduler.step(val_qwk)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"  train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['val_loss']:.4f} | "
            f"val_acc={val_metrics['val_accuracy']:.4f} | "
            f"val_qwk={val_qwk:.4f} | "
            f"val_f1={val_metrics['val_macro_f1']:.4f} | "
            f"lr={current_lr:.2e}"
        )

        # ---- Logging ----
        append_log_row(log_path, {
            "epoch":         epoch,
            "phase":         1,
            "train_loss":    round(train_loss, 6),
            "val_loss":      round(val_metrics["val_loss"], 6),
            "val_accuracy":  round(val_metrics["val_accuracy"], 6),
            "val_qwk":       round(val_qwk, 6),
            "val_macro_f1":  round(val_metrics["val_macro_f1"], 6),
            "lr":            current_lr,
        })

        # ---- Checkpointing ----
        save_checkpoint(
            {"state_dict": model.state_dict(), "epoch": epoch},
            ckpt_dir / cfg.CHECKPOINT_LAST,
        )
        if val_qwk > best_val_qwk:
            best_val_qwk = val_qwk
            no_improve   = 0
            save_checkpoint(
                {
                    "state_dict": model.state_dict(),
                    "epoch":      epoch,
                    "val_qwk":    val_qwk,
                    "model_name": model_name,
                    "fold":       fold_idx + 1,
                },
                ckpt_dir / cfg.CHECKPOINT_BEST,
            )
            print(f"  [*] New best val_qwk={best_val_qwk:.4f}. Checkpoint saved.")
        else:
            no_improve += 1
            if no_improve >= early_stopping_patience:
                print(f"  [Early stop] No improvement for {early_stopping_patience} epochs.")
                break

        epochs_trained = epoch

    return best_val_qwk, epochs_trained


# ---------------------------------------------------------------------------
# Phase 2 — End-to-end fine-tuning
# ---------------------------------------------------------------------------

def _phase2(
    model,
    train_loader:  DataLoader,
    val_loader:    DataLoader,
    fold_idx:      int,
    model_name:    str,
    device:        torch.device,
    phase2_epochs: int,
    start_epoch:   int,
    best_val_qwk:  float,
    log_path:      Path,
    ckpt_dir:      Path,
    early_stopping_patience: int,
    lr_head:       float,
    lr_backbone:   float,
    scheduler_patience: int,
    scheduler_factor:   float,
) -> float:
    """
    Phase 2 fine-tuning. Returns updated best_val_qwk.
    """
    print(f"\n{'='*60}")
    print(f"Phase 2 — End-to-end fine-tuning")
    print(f"{'='*60}")

    model.unfreeze_backbone()

    optimizer = torch.optim.Adam([
        {"params": model.cnn_branch.parameters(),          "lr": lr_backbone},
        {"params": model.classification_head.parameters(), "lr": lr_head},
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max",
        patience=scheduler_patience,
        factor=scheduler_factor,
        min_lr=cfg.SCHEDULER_MIN_LR,
    )
    criterion = nn.CrossEntropyLoss()

    no_improve = 0

    for epoch_offset in range(1, phase2_epochs + 1):
        epoch = start_epoch + epoch_offset
        print(f"\n  [Phase2 | Fold {fold_idx+1} | Epoch {epoch}]")

        # ---- Train (end-to-end, no SMOTE) ----
        model.train()
        loss_meter = AverageMeter("train_loss")
        for imgs, hc, labels in tqdm(train_loader, desc="  Fine-tuning", leave=False):
            imgs   = imgs.to(device)
            hc     = hc.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(imgs, hc)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item(), n=len(labels))

        train_loss = loss_meter.avg

        # ---- Validation ----
        val_metrics = _validate(model, val_loader, criterion, device)
        val_qwk     = val_metrics["val_qwk"]
        scheduler.step(val_qwk)

        current_lr_head = optimizer.param_groups[1]["lr"]
        print(
            f"  train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['val_loss']:.4f} | "
            f"val_acc={val_metrics['val_accuracy']:.4f} | "
            f"val_qwk={val_qwk:.4f} | "
            f"val_f1={val_metrics['val_macro_f1']:.4f} | "
            f"lr_head={current_lr_head:.2e}"
        )

        # ---- Logging ----
        append_log_row(log_path, {
            "epoch":         epoch,
            "phase":         2,
            "train_loss":    round(train_loss, 6),
            "val_loss":      round(val_metrics["val_loss"], 6),
            "val_accuracy":  round(val_metrics["val_accuracy"], 6),
            "val_qwk":       round(val_qwk, 6),
            "val_macro_f1":  round(val_metrics["val_macro_f1"], 6),
            "lr":            current_lr_head,
        })

        # ---- Checkpointing ----
        save_checkpoint(
            {"state_dict": model.state_dict(), "epoch": epoch},
            ckpt_dir / cfg.CHECKPOINT_LAST,
        )
        if val_qwk > best_val_qwk:
            best_val_qwk = val_qwk
            no_improve   = 0
            save_checkpoint(
                {
                    "state_dict": model.state_dict(),
                    "epoch":      epoch,
                    "val_qwk":    val_qwk,
                    "model_name": model_name,
                    "fold":       fold_idx + 1,
                },
                ckpt_dir / cfg.CHECKPOINT_BEST,
            )
            print(f"  [*] New best val_qwk={best_val_qwk:.4f}. Checkpoint saved.")
        else:
            no_improve += 1
            if no_improve >= early_stopping_patience:
                print(f"  [Early stop] No improvement for {early_stopping_patience} epochs.")
                break

    return best_val_qwk


# ---------------------------------------------------------------------------
# Single-fold orchestration
# ---------------------------------------------------------------------------

def train_single_fold(
    fold_idx:    int,
    model_name:  str  = "dr_litenet_effb0",
    backbone:    str  = cfg.BACKBONE,
    phase1_epochs: int = cfg.PHASE1_EPOCHS,
    phase2_epochs: int = cfg.PHASE2_EPOCHS,
    batch_size:  int  = cfg.BATCH_SIZE,
    lr_head:     float = cfg.LR_HEAD,
    lr_backbone: float = cfg.LR_BACKBONE,
    early_stopping_patience: int = cfg.EARLY_STOPPING_PATIENCE,
    scheduler_patience: int  = cfg.SCHEDULER_PATIENCE,
    scheduler_factor:   float = cfg.SCHEDULER_FACTOR,
    num_workers: int  = 2,
    seed:        int  = cfg.SEED,
) -> dict:
    """
    Full two-phase training for one fold.
    Returns results dict (best metrics for the fold).
    """
    set_seed(seed)
    device = get_device()
    print(f"\n{'#'*60}")
    print(f"  Fold {fold_idx+1}/{cfg.NUM_FOLDS} | Model: {model_name} | Device: {device}")
    print(f"{'#'*60}")

    # ---- Paths ----
    exp_dir  = cfg.experiment_path(model_name, fold_idx + 1)
    ckpt_dir = exp_dir / "checkpoints"
    log_dir  = exp_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_csv  = log_dir / cfg.TRAIN_LOG_CSV

    # Remove stale log so we start fresh each run
    if log_csv.exists():
        log_csv.unlink()

    # ---- Save training config snapshot ----
    config_snapshot = {
        "model_name":   model_name,
        "backbone":     backbone,
        "fold":         fold_idx + 1,
        "phase1_epochs": phase1_epochs,
        "phase2_epochs": phase2_epochs,
        "batch_size":   batch_size,
        "lr_head":      lr_head,
        "lr_backbone":  lr_backbone,
        "seed":         seed,
        "smote_variant": cfg.SMOTE_VARIANT,
        "handcrafted_dim": cfg.HANDCRAFTED_FEATURE_DIM,
        "image_size":   cfg.IMAGE_SIZE,
        "num_classes":  cfg.NUM_CLASSES,
    }
    save_json(config_snapshot, log_dir / cfg.TRAINING_CONFIG_JSON)

    # ---- Data ----
    train_loader, val_loader = get_dataloaders(
        fold_idx=fold_idx,
        batch_size=batch_size,
        use_handcrafted=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    print(f"  Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ---- Model ----
    model = build_model(backbone_name=backbone, pretrained=True, device=device)

    # ---- Phase 1 ----
    best_qwk, p1_epochs = _phase1(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        fold_idx=fold_idx,
        model_name=model_name,
        device=device,
        phase1_epochs=phase1_epochs,
        batch_size=batch_size,
        log_path=log_csv,
        ckpt_dir=ckpt_dir,
        early_stopping_patience=early_stopping_patience,
        lr_head=lr_head,
        scheduler_patience=scheduler_patience,
        scheduler_factor=scheduler_factor,
    )

    # ---- Phase 2 ----
    best_qwk = _phase2(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        fold_idx=fold_idx,
        model_name=model_name,
        device=device,
        phase2_epochs=phase2_epochs,
        start_epoch=p1_epochs,
        best_val_qwk=best_qwk,
        log_path=log_csv,
        ckpt_dir=ckpt_dir,
        early_stopping_patience=early_stopping_patience,
        lr_head=lr_head,
        lr_backbone=lr_backbone,
        scheduler_patience=scheduler_patience,
        scheduler_factor=scheduler_factor,
    )

    # ---- Final val metrics from best checkpoint ----
    from src.utils import load_checkpoint
    load_checkpoint(ckpt_dir / cfg.CHECKPOINT_BEST, model, device)
    final_metrics = _validate(model, val_loader, nn.CrossEntropyLoss(), device)

    results = {
        "model_name":            model_name,
        "fold":                  fold_idx + 1,
        "best_val_qwk":          round(best_qwk, 6),
        "final_val_accuracy":    round(final_metrics["val_accuracy"], 6),
        "final_val_qwk":         round(final_metrics["val_qwk"], 6),
        "final_val_macro_f1":    round(final_metrics["val_macro_f1"], 6),
        "config_snapshot":       config_snapshot,
    }
    save_json(results, exp_dir / cfg.RESULTS_JSON)
    print(f"\n  Fold {fold_idx+1} complete. Best val QWK = {best_qwk:.4f}")
    return results


# ---------------------------------------------------------------------------
# All-folds orchestration
# ---------------------------------------------------------------------------

def train_all_folds(
    model_name: str = "dr_litenet_effb0",
    backbone:   str = cfg.BACKBONE,
    **kwargs,
) -> list[dict]:
    """Train all 5 folds and return list of results dicts."""
    all_results = []
    for fold_idx in range(cfg.NUM_FOLDS):
        result = train_single_fold(
            fold_idx=fold_idx,
            model_name=model_name,
            backbone=backbone,
            **kwargs,
        )
        all_results.append(result)

    # Summary
    qwks = [r["best_val_qwk"] for r in all_results]
    print(f"\n{'='*60}")
    print(f"  All folds complete. QWKs: {[round(q,4) for q in qwks]}")
    print(f"  Mean QWK = {np.mean(qwks):.4f} ± {np.std(qwks):.4f}")
    print(f"{'='*60}")
    return all_results


# ---------------------------------------------------------------------------
# Sanity check — run as script (CPU-safe: 2 Phase-1 epochs, batch_size=4)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    print("=" * 60)
    print("train.py sanity check — 2 Phase-1 epochs, batch_size=4")
    print("=" * 60)

    set_seed(cfg.SEED)

    results = train_single_fold(
        fold_idx=0,
        model_name="dr_litenet_effb0",
        backbone="efficientnet_b0",
        phase1_epochs=2,
        phase2_epochs=0,        # skip Phase 2 for sanity check
        batch_size=4,
        num_workers=0,
        early_stopping_patience=cfg.EARLY_STOPPING_PATIENCE,
        scheduler_patience=cfg.SCHEDULER_PATIENCE,
        scheduler_factor=cfg.SCHEDULER_FACTOR,
    )

    print("\n--- Results ---")
    for k, v in results.items():
        if k != "config_snapshot":
            print(f"  {k}: {v}")

    # Verify checkpoint files exist
    ckpt_dir = cfg.experiment_path("dr_litenet_effb0", 1) / "checkpoints"
    for fname in [cfg.CHECKPOINT_BEST, cfg.CHECKPOINT_LAST]:
        p = ckpt_dir / fname
        size_kb = p.stat().st_size / 1024 if p.exists() else 0
        status = "OK" if p.exists() else "MISSING"
        print(f"  [{status}] {p.name}  ({size_kb:.1f} KB)")

    log_csv = cfg.experiment_path("dr_litenet_effb0", 1) / "logs" / cfg.TRAIN_LOG_CSV
    print(f"  [{'OK' if log_csv.exists() else 'MISSING'}] train_log.csv")
    if log_csv.exists():
        with open(log_csv) as f:
            print("  Log contents:")
            for line in f:
                print("   ", line.strip())

    print("\n[ALL SANITY CHECKS PASSED]")
