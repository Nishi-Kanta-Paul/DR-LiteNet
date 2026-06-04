"""
Central hyperparameter registry for DR-LiteNet.
All training constants, paths, and flags live here.
Import this module everywhere; never hardcode values in other files.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple

# ---------------------------------------------------------------------------
# Root paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Data paths  (actual layout discovered in repo — nested image folders)
# ---------------------------------------------------------------------------
DATA_ROOT = PROJECT_ROOT / "data"

APTOS_TRAIN_CSV   = DATA_ROOT / "train_1.csv"          # 2929 rows; columns: id_code, diagnosis
APTOS_VALID_CSV   = DATA_ROOT / "valid.csv"            # 365 rows
APTOS_TEST_CSV    = DATA_ROOT / "test.csv"             # 868 rows
APTOS_TRAIN_DIR   = DATA_ROOT / "train_images" / "train_images"
APTOS_VAL_DIR     = DATA_ROOT / "val_images"   / "val_images"
APTOS_TEST_DIR    = DATA_ROOT / "test_images"  / "test_images"

IDRID_CSV         = DATA_ROOT / "idrid" / "labels.csv"  # not yet available
IDRID_IMAGE_DIR   = DATA_ROOT / "idrid" / "images"      # not yet available

PROCESSED_DIR     = DATA_ROOT / "processed"

# ---------------------------------------------------------------------------
# Experiment / output paths
# ---------------------------------------------------------------------------
EXPERIMENT_DIR    = PROJECT_ROOT / "experiments"
OUTPUTS_DIR       = PROJECT_ROOT / "outputs"
FIGURES_DIR       = OUTPUTS_DIR  / "figures"
GRAD_CAM_DIR      = FIGURES_DIR  / "grad_cam"
TABLES_DIR        = OUTPUTS_DIR  / "tables"
PREDICTIONS_DIR   = OUTPUTS_DIR  / "predictions"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED: int = 42

# ---------------------------------------------------------------------------
# Dataset / preprocessing
# ---------------------------------------------------------------------------
IMAGE_SIZE: int = 224                          # pixels (both H and W)
NUM_CLASSES: int = 5
NUM_FOLDS: int = 5                             # stratified k-fold on APTOS train set

CLASS_NAMES: Dict[int, str] = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR",
}

# CLAHE (applied on L-channel in LAB colour space)
CLAHE_CLIP_LIMIT: float = 2.0
CLAHE_TILE_GRID: Tuple[int, int] = (8, 8)

# Gaussian blur
GAUSSIAN_KERNEL: Tuple[int, int] = (3, 3)
GAUSSIAN_SIGMA: float = 1.0

# ImageNet normalisation (backbone pretrained on ImageNet)
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD:  Tuple[float, float, float] = (0.229, 0.224, 0.225)

# Augmentation flags (training set only)
AUG_HFLIP: bool = True
AUG_VFLIP: bool = True
AUG_ROTATION_DEG: int = 15
AUG_COLOR_JITTER: bool = True
AUG_JITTER_BRIGHTNESS: float = 0.2
AUG_JITTER_CONTRAST: float = 0.2
AUG_JITTER_SATURATION: float = 0.2

# ---------------------------------------------------------------------------
# Handcrafted feature branch
# ---------------------------------------------------------------------------
# Colour histogram: 16 bins per channel in HSV → 48-d
# Scalars: vessel_density (1), exudate_intensity (1), texture_entropy (1), placeholder (1)
COLOR_HIST_BINS: int = 16                      # per HSV channel
HANDCRAFTED_FEATURE_DIM: int = 52             # 3 × 16 (hist) + 4 scalars

# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------
BACKBONE: str = "efficientnet_b0"             # options: "efficientnet_b0" | "mobilenet_v3_small"
PRETRAINED: bool = True

# CNN branch expected GAP output dims (validated by dummy forward pass in model.py)
BACKBONE_OUTPUT_DIMS: Dict[str, int] = {
    "efficientnet_b0":    1280,
    "mobilenet_v3_small":  576,
}

HIDDEN_DIM: int = 512                         # dense head hidden layer width
DROPOUT_RATE: float = 0.4

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BATCH_SIZE: int = 32
BATCH_SIZE_PHASE2: int = 16                   # fine-tuning phase (lower for memory safety)
MAX_EPOCHS: int = 50
EARLY_STOPPING_PATIENCE: int = 10            # epochs, monitored on val QWK

OPTIMIZER: str = "adam"
LR_HEAD: float = 1e-4                         # dense head / Phase 1
LR_BACKBONE: float = 1e-5                     # backbone fine-tuning / Phase 2

# LR scheduler
SCHEDULER: str = "ReduceLROnPlateau"
SCHEDULER_PATIENCE: int = 5
SCHEDULER_FACTOR: float = 0.5
SCHEDULER_MODE: str = "max"                   # monitor val QWK (higher is better)
SCHEDULER_MIN_LR: float = 1e-7

# Two-phase training flags
PHASE1_EPOCHS: int = 20                       # frozen backbone → SMOTE → head training
PHASE2_EPOCHS: int = 30                       # backbone fine-tune end-to-end

# Mixed precision (set True on GPU to reduce memory)
USE_AMP: bool = False

# ---------------------------------------------------------------------------
# Adaptive SMOTE
# ---------------------------------------------------------------------------
SMOTE_VARIANT: str = "ADASYN"                 # options: "ADASYN" | "SMOTE" | "BorderlineSMOTE"
SMOTE_SAMPLING_STRATEGY: str = "auto"        # oversample all minority classes to majority count
SMOTE_RANDOM_STATE: int = SEED
SMOTE_N_NEIGHBORS: int = 5

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
PRIMARY_METRIC: str = "qwk"                   # model selection metric per fold

# ---------------------------------------------------------------------------
# Experiment naming
# ---------------------------------------------------------------------------
MODEL_NAMES = [
    "dr_litenet_effb0",
    "dr_litenet_mobv3",
    "effb0_baseline",
    "effb0_weighted",
    "mobv3_baseline",
    "resnet50_baseline",
    "dr_litenet_no_smote",
    "dr_litenet_weighted",
]

CHECKPOINT_BEST = "best_model.pth"
CHECKPOINT_LAST = "last_model.pth"
TRAIN_LOG_CSV   = "train_log.csv"
TRAINING_CONFIG_JSON = "training_config.json"
RESULTS_JSON    = "results.json"


# ---------------------------------------------------------------------------
# Helper: return the experiment directory for a given model + fold
# ---------------------------------------------------------------------------
def experiment_path(model_name: str, fold: int) -> Path:
    return EXPERIMENT_DIR / model_name / f"fold_{fold}"


def checkpoint_path(model_name: str, fold: int, filename: str = CHECKPOINT_BEST) -> Path:
    return experiment_path(model_name, fold) / "checkpoints" / filename


def log_path(model_name: str, fold: int) -> Path:
    return experiment_path(model_name, fold) / "logs"


# ---------------------------------------------------------------------------
# Sanity-check print (run this file directly to verify all values)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    cfg = {
        "PROJECT_ROOT":          str(PROJECT_ROOT),
        "APTOS_TRAIN_CSV":       str(APTOS_TRAIN_CSV),
        "APTOS_VALID_CSV":       str(APTOS_VALID_CSV),
        "APTOS_TEST_CSV":        str(APTOS_TEST_CSV),
        "APTOS_TRAIN_DIR":       str(APTOS_TRAIN_DIR),
        "IMAGE_SIZE":            IMAGE_SIZE,
        "NUM_CLASSES":           NUM_CLASSES,
        "CLASS_NAMES":           CLASS_NAMES,
        "CLAHE_CLIP_LIMIT":      CLAHE_CLIP_LIMIT,
        "CLAHE_TILE_GRID":       CLAHE_TILE_GRID,
        "HANDCRAFTED_FEATURE_DIM": HANDCRAFTED_FEATURE_DIM,
        "BACKBONE":              BACKBONE,
        "BACKBONE_OUTPUT_DIMS":  BACKBONE_OUTPUT_DIMS,
        "HIDDEN_DIM":            HIDDEN_DIM,
        "DROPOUT_RATE":          DROPOUT_RATE,
        "BATCH_SIZE":            BATCH_SIZE,
        "BATCH_SIZE_PHASE2":     BATCH_SIZE_PHASE2,
        "MAX_EPOCHS":            MAX_EPOCHS,
        "LR_HEAD":               LR_HEAD,
        "LR_BACKBONE":           LR_BACKBONE,
        "EARLY_STOPPING_PATIENCE": EARLY_STOPPING_PATIENCE,
        "SCHEDULER":             SCHEDULER,
        "SCHEDULER_PATIENCE":    SCHEDULER_PATIENCE,
        "SCHEDULER_FACTOR":      SCHEDULER_FACTOR,
        "SMOTE_VARIANT":         SMOTE_VARIANT,
        "SMOTE_SAMPLING_STRATEGY": SMOTE_SAMPLING_STRATEGY,
        "SEED":                  SEED,
        "NUM_FOLDS":             NUM_FOLDS,
        "PRIMARY_METRIC":        PRIMARY_METRIC,
    }

    print(json.dumps(cfg, indent=2))

    # Path existence checks
    print("\n--- Data path checks ---")
    for label, p in [
        ("APTOS_TRAIN_CSV",  APTOS_TRAIN_CSV),
        ("APTOS_VALID_CSV",  APTOS_VALID_CSV),
        ("APTOS_TEST_CSV",   APTOS_TEST_CSV),
        ("APTOS_TRAIN_DIR",  APTOS_TRAIN_DIR),
        ("APTOS_VAL_DIR",    APTOS_VAL_DIR),
        ("APTOS_TEST_DIR",   APTOS_TEST_DIR),
    ]:
        status = "OK" if Path(p).exists() else "MISSING"
        print(f"  [{status}] {label}: {p}")
