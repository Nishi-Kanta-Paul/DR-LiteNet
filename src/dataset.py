"""
Dataset loading, preprocessing, handcrafted feature extraction,
fold splitting, and DataLoader factory for DR-LiteNet.

Handles: APTOS 2019 (primary) and IDRiD (external validation, when available).
"""

import sys
from pathlib import Path

# Allow `python src/dataset.py` from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from scipy.stats import entropy as scipy_entropy

import src.config as cfg


# ---------------------------------------------------------------------------
# 1. Preprocessing helpers
# ---------------------------------------------------------------------------

def _apply_clahe(bgr: np.ndarray) -> np.ndarray:
    """CLAHE on L-channel (LAB colour space). Returns BGR uint8."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=cfg.CLAHE_CLIP_LIMIT,
        tileGridSize=cfg.CLAHE_TILE_GRID,
    )
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def _circular_crop(bgr: np.ndarray) -> np.ndarray:
    """
    Detect the retinal disc boundary, crop to its bounding square,
    and mask outside pixels to black.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    # Find the largest contour (the retinal disc)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return bgr

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    # Make the crop square around the disc centre
    cx, cy = x + w // 2, y + h // 2
    r = max(w, h) // 2
    x1 = max(0, cx - r)
    y1 = max(0, cy - r)
    x2 = min(bgr.shape[1], cx + r)
    y2 = min(bgr.shape[0], cy + r)

    cropped = bgr[y1:y2, x1:x2]

    # Mask pixels outside the circle
    mask = np.zeros(cropped.shape[:2], dtype=np.uint8)
    cr, cc = cropped.shape[0] // 2, cropped.shape[1] // 2
    radius = min(cr, cc)
    cv2.circle(mask, (cc, cr), radius, 255, -1)
    cropped = cv2.bitwise_and(cropped, cropped, mask=mask)

    return cropped


def preprocess_image(bgr: np.ndarray) -> np.ndarray:
    """
    Full preprocessing pipeline:
      CLAHE → circular crop → Gaussian blur → resize 224×224.
    Returns uint8 RGB numpy array (H, W, 3).
    Note: ImageNet normalisation is applied later by the torchvision transform.
    """
    img = _apply_clahe(bgr)
    img = _circular_crop(img)
    img = cv2.GaussianBlur(img, cfg.GAUSSIAN_KERNEL, cfg.GAUSSIAN_SIGMA)
    img = cv2.resize(img, (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# ---------------------------------------------------------------------------
# 2. Torchvision transforms
# ---------------------------------------------------------------------------

def _base_transform() -> transforms.Compose:
    """Resize + ToTensor + ImageNet normalise. Applied to every split."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.IMAGENET_MEAN, std=cfg.IMAGENET_STD),
    ])


def _train_transform() -> transforms.Compose:
    """Augmentation + base transform. Applied to training set only."""
    aug = []
    if cfg.AUG_HFLIP:
        aug.append(transforms.RandomHorizontalFlip())
    if cfg.AUG_VFLIP:
        aug.append(transforms.RandomVerticalFlip())
    aug.append(transforms.RandomRotation(cfg.AUG_ROTATION_DEG))
    if cfg.AUG_COLOR_JITTER:
        aug.append(transforms.ColorJitter(
            brightness=cfg.AUG_JITTER_BRIGHTNESS,
            contrast=cfg.AUG_JITTER_CONTRAST,
            saturation=cfg.AUG_JITTER_SATURATION,
        ))
    aug += [
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.IMAGENET_MEAN, std=cfg.IMAGENET_STD),
    ]
    return transforms.Compose(aug)


# ---------------------------------------------------------------------------
# 3. Handcrafted feature extraction
# ---------------------------------------------------------------------------

def extract_handcrafted_features(rgb: np.ndarray) -> np.ndarray:
    """
    Extract the 52-dimensional handcrafted lesion descriptor vector from
    a preprocessed RGB image (uint8, 224×224).

    Layout: [vessel_density, exudate_intensity, texture_entropy, 0.0, *hist_48d]
      - Index 0:  vessel density (scalar)
      - Index 1:  exudate intensity (scalar)
      - Index 2:  texture entropy (scalar)
      - Index 3:  reserved / zero-padded (keeps 4 scalar slots)
      - Index 4–51: HSV colour histogram (3 channels × 16 bins, normalised)
    """
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # ------------------------------------------------------------------
    # 3a. Vessel density — morphological green-channel filtering
    # ------------------------------------------------------------------
    green = bgr[:, :, 1]

    # Top-hat transform to enhance thin vessel-like structures
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tophat = cv2.morphologyEx(green, cv2.MORPH_TOPHAT, kernel)

    # Threshold to get binary vessel mask
    _, vessel_mask = cv2.threshold(tophat, 10, 255, cv2.THRESH_BINARY)

    # Retinal area mask (non-black pixels)
    _, retinal_mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    retinal_area = float(np.count_nonzero(retinal_mask))
    vessel_pixels = float(np.count_nonzero(vessel_mask))
    vessel_density = vessel_pixels / retinal_area if retinal_area > 0 else 0.0

    # ------------------------------------------------------------------
    # 3b. Exudate intensity — bright-lesion threshold on CLAHE green ch
    # ------------------------------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=cfg.CLAHE_CLIP_LIMIT, tileGridSize=cfg.CLAHE_TILE_GRID)
    green_eq = clahe.apply(green)

    # Bright lesions: pixels above 90th percentile within the retinal area
    retinal_pixels = green_eq[retinal_mask > 0]
    if retinal_pixels.size > 0:
        thresh_val = float(np.percentile(retinal_pixels, 90))
        exudate_mask = (green_eq > thresh_val) & (retinal_mask > 0)
        exudate_vals = green_eq[exudate_mask]
        exudate_intensity = float(np.mean(exudate_vals)) / 255.0 if exudate_vals.size > 0 else 0.0
    else:
        exudate_intensity = 0.0

    # ------------------------------------------------------------------
    # 3c. Texture entropy — Shannon entropy over grayscale histogram
    # ------------------------------------------------------------------
    hist_gray, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256), density=True)
    hist_gray = hist_gray + 1e-10  # avoid log(0)
    texture_entropy = float(scipy_entropy(hist_gray))
    # Normalise to [0,1] range (max entropy for 256 bins = log(256) ≈ 5.55)
    texture_entropy = texture_entropy / np.log(256)

    # ------------------------------------------------------------------
    # 3d. Colour histogram — HSV, 16 bins/channel, normalised
    # ------------------------------------------------------------------
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hist_features = []
    ranges = [(0, 180), (0, 256), (0, 256)]   # H, S, V ranges in OpenCV
    for ch in range(3):
        h, _ = np.histogram(
            hsv[:, :, ch].ravel(),
            bins=cfg.COLOR_HIST_BINS,
            range=ranges[ch],
        )
        h = h.astype(np.float32)
        total = h.sum()
        if total > 0:
            h /= total
        hist_features.append(h)
    hist_vec = np.concatenate(hist_features, axis=0)  # 48-d

    # ------------------------------------------------------------------
    # Assemble final 52-d vector
    # ------------------------------------------------------------------
    scalars = np.array(
        [vessel_density, exudate_intensity, texture_entropy, 0.0],
        dtype=np.float32,
    )
    features = np.concatenate([scalars, hist_vec], axis=0)  # (52,)
    assert features.shape == (cfg.HANDCRAFTED_FEATURE_DIM,), \
        f"Expected {cfg.HANDCRAFTED_FEATURE_DIM}-d, got {features.shape}"
    return features


# ---------------------------------------------------------------------------
# 4. PyTorch Dataset
# ---------------------------------------------------------------------------

class APTOSDataset(Dataset):
    """
    PyTorch Dataset for APTOS 2019 (and IDRiD with same CSV format).

    Args:
        image_dir:   Path to folder containing .png / .jpg images.
        csv_path:    Path to CSV with columns [id_code, diagnosis].
        indices:     Optional list of row indices into the CSV (for fold splitting).
                     If None, the full CSV is used.
        augment:     If True, apply training augmentation.
        use_handcrafted: If True, extract and return 52-d handcrafted features.
        preprocess:  If True, apply the full preprocessing pipeline (CLAHE etc.).
    """

    def __init__(
        self,
        image_dir: Path,
        csv_path: Path,
        indices=None,
        augment: bool = False,
        use_handcrafted: bool = True,
        preprocess: bool = True,
    ):
        self.image_dir = Path(image_dir)
        self.augment = augment
        self.use_handcrafted = use_handcrafted
        self.preprocess = preprocess

        df = pd.read_csv(csv_path)
        if indices is not None:
            df = df.iloc[indices].reset_index(drop=True)

        self.image_ids = df["id_code"].tolist()
        self.labels = df["diagnosis"].tolist()

        self.transform = _train_transform() if augment else _base_transform()

    def __len__(self) -> int:
        return len(self.image_ids)

    def _load_image(self, image_id: str) -> np.ndarray:
        # Try .png first, then .jpg
        for ext in (".png", ".jpg", ".jpeg"):
            p = self.image_dir / f"{image_id}{ext}"
            if p.exists():
                bgr = cv2.imread(str(p))
                if bgr is not None:
                    return bgr
        raise FileNotFoundError(f"Image not found: {self.image_dir / image_id}.*")

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        label = self.labels[idx]

        bgr = self._load_image(image_id)

        # Preprocessing
        if self.preprocess:
            rgb = preprocess_image(bgr)
        else:
            rgb = cv2.cvtColor(
                cv2.resize(bgr, (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
                cv2.COLOR_BGR2RGB,
            )

        # Handcrafted features (extracted on uint8 preprocessed RGB)
        if self.use_handcrafted:
            hc_features = extract_handcrafted_features(rgb)
            hc_tensor = torch.from_numpy(hc_features).float()
        else:
            hc_tensor = torch.zeros(cfg.HANDCRAFTED_FEATURE_DIM, dtype=torch.float32)

        # torchvision transforms expect a PIL Image
        pil_img = Image.fromarray(rgb)
        img_tensor = self.transform(pil_img)

        return img_tensor, hc_tensor, torch.tensor(label, dtype=torch.long)


# ---------------------------------------------------------------------------
# 5. Fold splitting
# ---------------------------------------------------------------------------

def get_fold_splits(
    csv_path: Path = cfg.APTOS_TRAIN_CSV,
    num_folds: int = cfg.NUM_FOLDS,
    seed: int = cfg.SEED,
):
    """
    5-fold StratifiedKFold over APTOS training CSV.
    Returns list of (train_indices, val_indices) tuples (0-based row indices).
    """
    df = pd.read_csv(csv_path)
    labels = df["diagnosis"].values
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    splits = [(tr.tolist(), va.tolist()) for tr, va in skf.split(np.zeros(len(labels)), labels)]
    return splits


# ---------------------------------------------------------------------------
# 6. DataLoader factory
# ---------------------------------------------------------------------------

def get_dataloaders(
    fold_idx: int,
    batch_size: int = cfg.BATCH_SIZE,
    use_handcrafted: bool = True,
    num_workers: int = 2,
    pin_memory: bool = False,
):
    """
    Build train and validation DataLoaders for a given fold index (0-based).

    Training set  = fold split of APTOS train_1.csv (with augmentation).
    Validation set = fixed APTOS valid.csv (no augmentation).

    Returns: (train_loader, val_loader)
    """
    splits = get_fold_splits()
    train_indices, _ = splits[fold_idx]

    train_ds = APTOSDataset(
        image_dir=cfg.APTOS_TRAIN_DIR,
        csv_path=cfg.APTOS_TRAIN_CSV,
        indices=train_indices,
        augment=True,
        use_handcrafted=use_handcrafted,
    )
    val_ds = APTOSDataset(
        image_dir=cfg.APTOS_VAL_DIR,
        csv_path=cfg.APTOS_VALID_CSV,
        indices=None,
        augment=False,
        use_handcrafted=use_handcrafted,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader


def get_test_loader(
    batch_size: int = cfg.BATCH_SIZE,
    use_handcrafted: bool = True,
    num_workers: int = 2,
):
    """DataLoader for the APTOS test split (held-out). No augmentation."""
    test_ds = APTOSDataset(
        image_dir=cfg.APTOS_TEST_DIR,
        csv_path=cfg.APTOS_TEST_CSV,
        indices=None,
        augment=False,
        use_handcrafted=use_handcrafted,
    )
    return DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )


# ---------------------------------------------------------------------------
# 7. Sanity check — run as script
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from collections import Counter

    print("=" * 60)
    print("APTOS 2019 class distribution (train_1.csv)")
    print("=" * 60)
    df_train = pd.read_csv(cfg.APTOS_TRAIN_CSV)
    dist = Counter(df_train["diagnosis"].tolist())
    for c in sorted(dist):
        print(f"  Class {c} ({cfg.CLASS_NAMES[c]:20s}): {dist[c]:5d} images")
    print(f"  Total: {len(df_train)}")

    print("\nAPTOS 2019 class distribution (valid.csv)")
    df_val = pd.read_csv(cfg.APTOS_VALID_CSV)
    dist_v = Counter(df_val["diagnosis"].tolist())
    for c in sorted(dist_v):
        print(f"  Class {c} ({cfg.CLASS_NAMES[c]:20s}): {dist_v[c]:5d} images")
    print(f"  Total: {len(df_val)}")

    # ------------------------------------------------------------------
    # Fold 1 split
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("5-Fold StratifiedKFold — Fold 1 (index 0)")
    print("=" * 60)
    splits = get_fold_splits()
    tr_idx, va_idx = splits[0]
    df_fold_tr = df_train.iloc[tr_idx]
    df_fold_va = df_train.iloc[va_idx]
    print(f"  Train samples: {len(tr_idx)}")
    print(f"  Val   samples: {len(va_idx)}")
    print("  Train class dist:", dict(sorted(Counter(df_fold_tr["diagnosis"].tolist()).items())))
    print("  Val   class dist:", dict(sorted(Counter(df_fold_va["diagnosis"].tolist()).items())))

    # ------------------------------------------------------------------
    # One batch from train loader (batch_size=4, CPU-safe)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("DataLoader batch shapes (batch_size=4, fold 0)")
    print("=" * 60)
    train_loader, val_loader = get_dataloaders(fold_idx=0, batch_size=4, num_workers=0)
    imgs, hc, labels = next(iter(train_loader))
    print(f"  images tensor  : {tuple(imgs.shape)}   expected (4, 3, 224, 224)")
    print(f"  handcrafted    : {tuple(hc.shape)}       expected (4, 52)")
    print(f"  labels         : {tuple(labels.shape)}            expected (4,)")
    print(f"  label values   : {labels.tolist()}")
    print(f"  image  min/max : {imgs.min():.3f} / {imgs.max():.3f}")
    print(f"  hc     min/max : {hc.min():.4f} / {hc.max():.4f}")

    # ------------------------------------------------------------------
    # Handcrafted feature values for one sample
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Handcrafted feature breakdown (first sample in batch)")
    print("=" * 60)
    s = hc[0]
    print(f"  vessel_density    : {s[0]:.6f}")
    print(f"  exudate_intensity : {s[1]:.6f}")
    print(f"  texture_entropy   : {s[2]:.6f}")
    print(f"  reserved          : {s[3]:.6f}")
    print(f"  hist H (16-d)     : {[round(float(v),4) for v in s[4:20]]}")
    print(f"  hist S (16-d)     : {[round(float(v),4) for v in s[20:36]]}")
    print(f"  hist V (16-d)     : {[round(float(v),4) for v in s[36:52]]}")
    assert s[0] > 0 or s[1] > 0 or s[2] > 0, "All scalar features are zero — extraction problem!"
    assert s[4:52].sum() > 0, "Colour histogram is all-zero — extraction problem!"
    print("  [OK] Non-zero features confirmed.")

    # ------------------------------------------------------------------
    # Save a sample preprocessed image
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Saving sample fundus image → outputs/figures/sample_fundus.png")
    print("=" * 60)
    cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Denormalise first sample for display
    mean = torch.tensor(cfg.IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(cfg.IMAGENET_STD).view(3, 1, 1)
    img_display = imgs[0] * std + mean
    img_display = img_display.permute(1, 2, 0).clamp(0, 1).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(img_display)
    axes[0].set_title(f"Preprocessed (label={labels[0].item()}: {cfg.CLASS_NAMES[labels[0].item()]})")
    axes[0].axis("off")

    axes[1].bar(range(52), s.numpy(), color="steelblue", width=0.8)
    axes[1].set_title("Handcrafted feature vector (52-d)")
    axes[1].set_xlabel("Feature index")
    axes[1].set_ylabel("Value")

    plt.tight_layout()
    out_path = cfg.FIGURES_DIR / "sample_fundus.png"
    plt.savefig(out_path, dpi=100)
    plt.close()
    print(f"  Saved: {out_path}")

    print("\n[ALL SANITY CHECKS PASSED]")
