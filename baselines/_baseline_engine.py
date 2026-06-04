"""
Shared baseline training engine.

All baseline scripts are thin wrappers that call train_baseline_fold().
BaselineModel wraps any torchvision backbone for pure-CNN (no handcrafted) baselines.
DR-LiteNet ablation baselines reuse src.model.DRLiteNet directly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm
from torch.utils.data import DataLoader
from tqdm import tqdm

import src.config as cfg
from src.dataset import get_dataloaders, get_fold_splits
from src.evaluate import evaluate_fold
from src.utils import (
    AverageMeter,
    append_log_row,
    compute_accuracy,
    compute_macro_f1,
    compute_qwk,
    get_device,
    save_checkpoint,
    save_json,
    set_seed,
)


# ---------------------------------------------------------------------------
# Pure-CNN baseline model (no handcrafted branch)
# ---------------------------------------------------------------------------

class BaselineModel(nn.Module):
    """
    Standard transfer-learning baseline. Wraps a torchvision backbone with
    a linear classifier. forward(images, hc) ignores `hc` so it is compatible
    with the shared DataLoader that always yields (img, hc, label).

    Exposes get_cnn_last_conv_layer() and freeze/unfreeze for Grad-CAM parity.
    """

    SUPPORTED = {
        "efficientnet_b0":    (tvm.efficientnet_b0,    "DEFAULT", 1280),
        "mobilenet_v3_small": (tvm.mobilenet_v3_small, "DEFAULT",  576),
        "resnet50":           (tvm.resnet50,           "DEFAULT", 2048),
    }

    def __init__(
        self,
        backbone_name: str,
        num_classes:   int = cfg.NUM_CLASSES,
        pretrained:    bool = True,
        dropout_rate:  float = cfg.DROPOUT_RATE,
    ):
        super().__init__()
        if backbone_name not in self.SUPPORTED:
            raise ValueError(f"Unsupported backbone '{backbone_name}'")

        model_fn, weights_key, out_dim = self.SUPPORTED[backbone_name]
        weights = weights_key if pretrained else None
        base = model_fn(weights=weights)

        if backbone_name in ("efficientnet_b0", "mobilenet_v3_small"):
            self.features = base.features
            self.pool     = base.avgpool
            self._last_conv: nn.Conv2d = self.features[-1][0]
        else:  # resnet50
            # Keep everything up to (not including) the final FC
            children = list(base.children())
            self.features = nn.Sequential(*children[:-2])   # up to layer4
            self.pool     = children[-2]                    # AdaptiveAvgPool2d
            # Last conv layer: last block of layer4
            self._last_conv = list(base.layer4[-1].children())[-3]

        self.head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(out_dim, num_classes),
        )
        self.out_dim      = out_dim
        self.backbone_name = backbone_name

        # Verify actual output dim
        with torch.no_grad():
            dummy = torch.zeros(1, 3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)
            actual = self._forward_features(dummy).shape[1]
        assert actual == out_dim, (
            f"Expected dim {out_dim}, got {actual} for {backbone_name}"
        )

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return torch.flatten(x, 1)

    def forward(self, images: torch.Tensor, hc: torch.Tensor = None) -> torch.Tensor:
        feats = self._forward_features(images)
        return self.head(feats)

    # Compatibility shims so evaluate_fold / Grad-CAM work identically
    def extract_features(self, images: torch.Tensor, hc: torch.Tensor = None):
        with torch.no_grad():
            return self._forward_features(images)

    def get_cnn_last_conv_layer(self) -> nn.Conv2d:
        return self._last_conv

    def freeze_backbone(self):
        for p in self.features.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.features.parameters():
            p.requires_grad = True

    def count_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Class-weight helper
# ---------------------------------------------------------------------------

def compute_class_weights(
    fold_idx: int,
    device:   torch.device,
) -> torch.Tensor:
    """
    Inverse-frequency class weights from the training fold labels.
    Returns a (num_classes,) float tensor on `device`.
    """
    import pandas as pd
    from collections import Counter

    splits   = get_fold_splits()
    tr_idx, _ = splits[fold_idx]
    df        = pd.read_csv(cfg.APTOS_TRAIN_CSV)
    labels    = df.iloc[tr_idx]["diagnosis"].tolist()
    counts    = Counter(labels)
    total     = sum(counts.values())
    weights   = torch.zeros(cfg.NUM_CLASSES, dtype=torch.float32)
    for c in range(cfg.NUM_CLASSES):
        weights[c] = total / (cfg.NUM_CLASSES * counts.get(c, 1))
    return weights.to(device)


# ---------------------------------------------------------------------------
# Validation pass (works for both BaselineModel and DRLiteNet)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _validate(model, val_loader, criterion, device) -> dict:
    model.eval()
    loss_m = AverageMeter()
    all_preds, all_labels = [], []
    for imgs, hc, labels in val_loader:
        imgs, hc, labels = imgs.to(device), hc.to(device), labels.to(device)
        logits = model(imgs, hc)
        loss   = criterion(logits, labels)
        loss_m.update(loss.item(), len(labels))
        all_preds.extend(logits.argmax(1).cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
    return {
        "val_loss":     loss_m.avg,
        "val_accuracy": compute_accuracy(all_labels, all_preds),
        "val_qwk":      compute_qwk(all_labels, all_preds),
        "val_macro_f1": compute_macro_f1(all_labels, all_preds),
    }


# ---------------------------------------------------------------------------
# Inference timing helper
# ---------------------------------------------------------------------------

def measure_inference_time(model, device, n_runs: int = 50) -> float:
    """Returns mean inference time per image in milliseconds."""
    model.eval()
    dummy_img = torch.randn(1, 3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, device=device)
    dummy_hc  = torch.zeros(1, cfg.HANDCRAFTED_FEATURE_DIM, device=device)
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            model(dummy_img, dummy_hc)
    # Timed runs
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            model(dummy_img, dummy_hc)
    elapsed = time.perf_counter() - start
    return (elapsed / n_runs) * 1000.0  # ms per image


# ---------------------------------------------------------------------------
# Core baseline training loop (single fold, no SMOTE)
# ---------------------------------------------------------------------------

def train_baseline_fold(
    fold_idx:        int,
    model_name:      str,
    model:           nn.Module,           # already built + on device
    device:          torch.device,
    use_class_weights: bool  = False,
    use_handcrafted: bool    = False,     # False for pure-CNN baselines
    num_epochs:      int     = cfg.MAX_EPOCHS,
    batch_size:      int     = cfg.BATCH_SIZE,
    lr_head:         float   = cfg.LR_HEAD,
    lr_backbone:     float   = cfg.LR_BACKBONE,
    early_stopping_patience: int  = cfg.EARLY_STOPPING_PATIENCE,
    scheduler_patience:      int  = cfg.SCHEDULER_PATIENCE,
    scheduler_factor:        float = cfg.SCHEDULER_FACTOR,
    num_workers:     int     = 2,
    seed:            int     = cfg.SEED,
) -> dict:
    """
    Standard end-to-end training (no SMOTE, no phase splitting).
    Works for both BaselineModel and DRLiteNet ablation variants.
    """
    set_seed(seed)

    # ---- Paths ----
    exp_dir  = cfg.experiment_path(model_name, fold_idx + 1)
    ckpt_dir = exp_dir / "checkpoints"
    log_dir  = exp_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_csv  = log_dir / cfg.TRAIN_LOG_CSV
    if log_csv.exists():
        log_csv.unlink()

    # ---- Config snapshot ----
    config_snapshot = {
        "model_name":        model_name,
        "fold":              fold_idx + 1,
        "num_epochs":        num_epochs,
        "batch_size":        batch_size,
        "use_class_weights": use_class_weights,
        "use_handcrafted":   use_handcrafted,
        "seed":              seed,
    }
    save_json(config_snapshot, log_dir / cfg.TRAINING_CONFIG_JSON)

    # ---- Data ----
    train_loader, val_loader = get_dataloaders(
        fold_idx=fold_idx,
        batch_size=batch_size,
        use_handcrafted=use_handcrafted,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # ---- Loss ----
    if use_class_weights:
        weights   = compute_class_weights(fold_idx, device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        print(f"  Class weights: {weights.cpu().numpy().round(3).tolist()}")
    else:
        criterion = nn.CrossEntropyLoss()

    # ---- Optimizer: differential LR (backbone vs head) ----
    head_params     = list(model.head.parameters()) \
                      if hasattr(model, "head") \
                      else list(model.classification_head.parameters())
    backbone_params = [p for p in model.parameters()
                       if not any(p is hp for hp in head_params)]
    optimizer = torch.optim.Adam([
        {"params": backbone_params, "lr": lr_backbone},
        {"params": head_params,     "lr": lr_head},
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max",
        patience=scheduler_patience,
        factor=scheduler_factor,
        min_lr=cfg.SCHEDULER_MIN_LR,
    )

    best_val_qwk = -1.0
    no_improve   = 0

    for epoch in range(1, num_epochs + 1):
        # ---- Train ----
        model.train()
        loss_m = AverageMeter()
        for imgs, hc, labels in tqdm(
            train_loader, desc=f"  [{model_name} | fold {fold_idx+1} | ep {epoch}]",
            leave=False,
        ):
            imgs, hc, labels = imgs.to(device), hc.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs, hc)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            loss_m.update(loss.item(), len(labels))

        # ---- Validate ----
        val_m   = _validate(model, val_loader, criterion, device)
        val_qwk = val_m["val_qwk"]
        scheduler.step(val_qwk)
        lr_now = optimizer.param_groups[1]["lr"]

        print(
            f"  ep{epoch:03d} train_loss={loss_m.avg:.4f} | "
            f"val_loss={val_m['val_loss']:.4f} | "
            f"val_acc={val_m['val_accuracy']:.4f} | "
            f"val_qwk={val_qwk:.4f} | "
            f"val_f1={val_m['val_macro_f1']:.4f} | lr={lr_now:.2e}"
        )

        # ---- Log ----
        append_log_row(log_csv, {
            "epoch":        epoch,
            "phase":        1,
            "train_loss":   round(loss_m.avg, 6),
            "val_loss":     round(val_m["val_loss"], 6),
            "val_accuracy": round(val_m["val_accuracy"], 6),
            "val_qwk":      round(val_qwk, 6),
            "val_macro_f1": round(val_m["val_macro_f1"], 6),
            "lr":           lr_now,
        })

        # ---- Checkpoint ----
        save_checkpoint({"state_dict": model.state_dict(), "epoch": epoch}, ckpt_dir / cfg.CHECKPOINT_LAST)
        if val_qwk > best_val_qwk:
            best_val_qwk = val_qwk
            no_improve   = 0
            save_checkpoint({
                "state_dict": model.state_dict(), "epoch": epoch,
                "val_qwk": val_qwk, "model_name": model_name, "fold": fold_idx + 1,
            }, ckpt_dir / cfg.CHECKPOINT_BEST)
            print(f"  [*] Best val_qwk={best_val_qwk:.4f}")
        else:
            no_improve += 1
            if no_improve >= early_stopping_patience:
                print(f"  [Early stop] after epoch {epoch}")
                break

    # ---- Full evaluation from best checkpoint ----
    from src.utils import load_checkpoint
    load_checkpoint(ckpt_dir / cfg.CHECKPOINT_BEST, model, device)
    result = evaluate_fold(
        model=model, val_loader=val_loader, device=device,
        fold_idx=fold_idx, model_name=model_name, save_results=True,
    )

    # Measure inference speed
    ms_per_img = measure_inference_time(model, device)
    param_count = sum(p.numel() for p in model.parameters())

    # Enrich results.json with param count + timing
    rpath = exp_dir / cfg.RESULTS_JSON
    import json
    with open(rpath) as f:
        saved = json.load(f)
    saved["param_count"]       = param_count
    saved["inference_time_ms"] = round(ms_per_img, 3)
    save_json(saved, rpath)

    print(f"\n  [{model_name} fold {fold_idx+1}] best QWK={best_val_qwk:.4f} | "
          f"params={param_count:,} | {ms_per_img:.1f} ms/img")
    return saved
