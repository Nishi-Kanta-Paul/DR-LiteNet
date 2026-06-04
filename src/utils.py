"""
Shared utilities: seed setting, device selection, metric helpers,
checkpoint I/O, feature caching, and training bookkeeping.
"""

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score, f1_score


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic ops where available (may slow down GPU training slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_qwk(y_true, y_pred) -> float:
    """Quadratic Weighted Kappa — primary evaluation metric."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(np.unique(y_true)) < 2:
        return 0.0
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))


def compute_macro_f1(y_true, y_pred) -> float:
    return float(f1_score(np.asarray(y_true), np.asarray(y_pred),
                          average="macro", zero_division=0))


def compute_accuracy(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).mean())


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(state: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: Path, model: torch.nn.Module, device: torch.device = None):
    """Load state_dict into model. Returns the full checkpoint dict."""
    if device is None:
        device = get_device()
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    return ckpt


# ---------------------------------------------------------------------------
# Feature caching  (npy files — avoids re-extracting features every epoch)
# ---------------------------------------------------------------------------

def cache_features(features: np.ndarray, labels: np.ndarray, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path) + "_X.npy", features)
    np.save(str(path) + "_y.npy", labels)


def load_cached_features(path: Path):
    X = np.load(str(path) + "_X.npy")
    y = np.load(str(path) + "_y.npy")
    return X, y


def cached_features_exist(path: Path) -> bool:
    return Path(str(path) + "_X.npy").exists() and Path(str(path) + "_y.npy").exists()


# ---------------------------------------------------------------------------
# CSV / JSON logging
# ---------------------------------------------------------------------------

def append_log_row(log_path: Path, row: dict) -> None:
    """Append one epoch's metrics as a CSV row (creates file + header if needed)."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.exists()
    with open(log_path, "a") as f:
        if write_header:
            f.write(",".join(row.keys()) + "\n")
        f.write(",".join(str(v) for v in row.values()) + "\n")


def save_json(data: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# AverageMeter — tracks running mean of a scalar (loss, accuracy, etc.)
# ---------------------------------------------------------------------------

class AverageMeter:
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count > 0 else 0.0

    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"
