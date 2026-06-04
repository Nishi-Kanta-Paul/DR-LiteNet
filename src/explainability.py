"""
Grad-CAM explainability for DR-LiteNet.

Classes:
  GradCAM  — hooks into target_layer, generates weighted heatmaps.

Functions:
  overlay_gradcam             — blend heatmap onto original image.
  create_gradcam_figure       — 3-panel matplotlib figure (orig | heatmap | overlay).
  create_gradcam_grid         — 5-class × N-sample grid figure.
  generate_gradcam_for_dataset — batch generation for correctly classified samples.
  generate_gradcam_misclassified — Grad-CAM for misclassified samples.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import src.config as cfg


# ---------------------------------------------------------------------------
# 1. GradCAM class
# ---------------------------------------------------------------------------

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.

    Usage:
        cam = GradCAM(model, target_layer=model.get_cnn_last_conv_layer())
        heatmap, pred_class, confidence = cam.generate(img_tensor, hc_tensor)
        cam.remove_hooks()
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model        = model
        self.target_layer = target_layer
        self._activations: Optional[torch.Tensor] = None
        self._gradients:   Optional[torch.Tensor] = None
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def _fwd_hook(module, input, output):
            self._activations = output.detach()   # (B, C, H, W)

        def _bwd_hook(module, grad_input, grad_output):
            self._gradients = grad_output[0].detach()  # (B, C, H, W)

        self._hooks.append(self.target_layer.register_forward_hook(_fwd_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(_bwd_hook))

    def remove_hooks(self):
        """Remove all registered hooks — call after generation to free memory."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def generate(
        self,
        images:               torch.Tensor,   # (1, 3, H, W)
        handcrafted_features: torch.Tensor,   # (1, 52)
        target_class:         Optional[int] = None,
    ) -> tuple[np.ndarray, int, float]:
        """
        Generate Grad-CAM heatmap for one image.

        Returns:
            heatmap      — (224, 224) float32 ndarray in [0, 1].
            pred_class   — predicted DR class index.
            confidence   — softmax confidence for the target class.
        """
        self.model.eval()

        images               = images.requires_grad_(False)
        handcrafted_features = handcrafted_features.requires_grad_(False)

        # Forward pass
        logits = self.model(images, handcrafted_features)   # (1, 5)
        probs  = F.softmax(logits, dim=1)

        pred_class = int(logits.argmax(dim=1).item())
        if target_class is None:
            target_class = pred_class
        confidence = float(probs[0, target_class].item())

        # Backward pass from target class logit
        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward()

        # Grad-CAM formula:  alpha_k = GAP(gradients)  →  heatmap = ReLU(sum alpha_k * A_k)
        activations = self._activations  # (1, C, H, W)
        gradients   = self._gradients    # (1, C, H, W)

        if activations is None or gradients is None:
            raise RuntimeError("Hooks did not fire — check target_layer registration.")

        # alpha_k: (1, C, 1, 1)
        alpha = gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted combination: (1, H, W)
        cam = (alpha * activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()                      # (H, W)

        # Normalise to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        # Resize to input spatial size
        heatmap = cv2.resize(cam, (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE))
        return heatmap.astype(np.float32), pred_class, confidence

    def __del__(self):
        self.remove_hooks()


# ---------------------------------------------------------------------------
# 2a. Heatmap overlay
# ---------------------------------------------------------------------------

def overlay_gradcam(
    original_rgb: np.ndarray,    # (H, W, 3) uint8 in [0,255]
    heatmap:      np.ndarray,    # (H, W) float32 in [0,1]
    alpha:        float = 0.45,
    colormap:     int   = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Returns uint8 RGB image: alpha-blended original + colour-mapped heatmap.
    """
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)        # BGR
    heatmap_rgb   = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)    # RGB

    blended = (alpha * heatmap_rgb.astype(np.float32)
               + (1 - alpha) * original_rgb.astype(np.float32))
    return blended.clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 2b. Single-sample 3-panel figure
# ---------------------------------------------------------------------------

def create_gradcam_figure(
    original_rgb:  np.ndarray,    # (H, W, 3) uint8
    heatmap:       np.ndarray,    # (H, W) float32 [0,1]
    predicted_class: int,
    true_class:      int,
    confidence:      float,
    image_id:        str = "",
) -> plt.Figure:
    """
    3-panel figure: original | heatmap | overlay.
    """
    overlay = overlay_gradcam(original_rgb, heatmap)
    pred_name = cfg.CLASS_NAMES[predicted_class]
    true_name = cfg.CLASS_NAMES[true_class]
    correct   = "✓" if predicted_class == true_class else "✗"

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(original_rgb)
    axes[0].set_title(f"Original\n(True: {true_name})", fontsize=9)
    axes[0].axis("off")

    im = axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM Heatmap", fontsize=9)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(overlay)
    axes[2].set_title(f"Overlay  {correct}\nPred: {pred_name} ({confidence:.1%})", fontsize=9)
    axes[2].axis("off")

    fig.suptitle(f"Grad-CAM  {image_id}", fontsize=10, y=1.01)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2c. 5-class × N-sample grid
# ---------------------------------------------------------------------------

def create_gradcam_grid(
    originals_by_class:  dict,   # {class_idx: [rgb_array, ...]}
    heatmaps_by_class:   dict,   # {class_idx: [heatmap, ...]}
    pred_classes_by_class: dict,
    confidences_by_class:  dict,
    samples_per_class:   int = 3,
    save_path:           Path = None,
) -> plt.Figure:
    """
    Grid: rows = DR severity classes, columns = samples.
    Each cell shows the overlay image with a small title.
    """
    n_rows = cfg.NUM_CLASSES
    n_cols = samples_per_class

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for row, cls in enumerate(range(cfg.NUM_CLASSES)):
        originals  = originals_by_class.get(cls, [])
        heatmaps   = heatmaps_by_class.get(cls, [])
        preds      = pred_classes_by_class.get(cls, [])
        confs      = confidences_by_class.get(cls, [])

        for col in range(n_cols):
            ax = axes[row, col]
            if col < len(originals):
                overlay = overlay_gradcam(originals[col], heatmaps[col])
                ax.imshow(overlay)
                correct = "✓" if preds[col] == cls else "✗"
                ax.set_title(
                    f"{correct} conf={confs[col]:.2f}\npred={cfg.CLASS_NAMES[preds[col]]}",
                    fontsize=7,
                )
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, color="gray")
            ax.axis("off")

        # Row label on leftmost cell
        axes[row, 0].set_ylabel(cfg.CLASS_NAMES[cls], fontsize=10, rotation=90,
                                labelpad=4)

    fig.suptitle("Grad-CAM Grid — All DR Severity Classes", fontsize=13, y=1.01)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=110, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Helper: denormalise tensor → uint8 RGB numpy
# ---------------------------------------------------------------------------

def _tensor_to_rgb(img_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a normalised (3, H, W) tensor back to (H, W, 3) uint8 RGB.
    """
    mean = torch.tensor(cfg.IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std  = torch.tensor(cfg.IMAGENET_STD,  dtype=torch.float32).view(3, 1, 1)
    rgb  = (img_tensor.cpu() * std + mean).clamp(0, 1)
    return (rgb.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 3. Batch generation — correctly classified samples
# ---------------------------------------------------------------------------

def generate_gradcam_for_dataset(
    model,
    val_loader:         DataLoader,
    device:             torch.device,
    num_samples_per_class: int  = 3,
    save_dir:           Path = None,
    model_name:         str  = "dr_litenet_effb0",
) -> dict:
    """
    For each DR class, find `num_samples_per_class` correctly classified images,
    generate Grad-CAM, save individual + grid figures.

    Returns dict: {class_idx: [{'image_id', 'heatmap', 'original', 'pred', 'conf'}]}
    """
    if save_dir is None:
        save_dir = cfg.GRAD_CAM_DIR
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    target_layer = model.get_cnn_last_conv_layer()
    cam          = GradCAM(model, target_layer)

    # Collect per-class results
    originals_by_class:   dict = {c: [] for c in range(cfg.NUM_CLASSES)}
    heatmaps_by_class:    dict = {c: [] for c in range(cfg.NUM_CLASSES)}
    preds_by_class:       dict = {c: [] for c in range(cfg.NUM_CLASSES)}
    confs_by_class:       dict = {c: [] for c in range(cfg.NUM_CLASSES)}
    needed = {c: num_samples_per_class for c in range(cfg.NUM_CLASSES)}

    saved_paths = []
    sample_idx  = 0

    for imgs, hc, labels in tqdm(val_loader, desc="  Grad-CAM (correct)", leave=False):
        if all(v == 0 for v in needed.values()):
            break

        for i in range(len(labels)):
            true_cls = int(labels[i].item())
            if needed.get(true_cls, 0) == 0:
                sample_idx += 1
                continue

            img_t = imgs[i:i+1].to(device)
            hc_t  = hc[i:i+1].to(device)

            heatmap, pred_cls, conf = cam.generate(img_t, hc_t, target_class=None)

            if pred_cls != true_cls:
                sample_idx += 1
                continue   # only correctly classified

            rgb = _tensor_to_rgb(imgs[i])
            originals_by_class[true_cls].append(rgb)
            heatmaps_by_class[true_cls].append(heatmap)
            preds_by_class[true_cls].append(pred_cls)
            confs_by_class[true_cls].append(conf)
            needed[true_cls] -= 1

            # Save individual figure
            fig = create_gradcam_figure(
                rgb, heatmap,
                predicted_class=pred_cls, true_class=true_cls,
                confidence=conf, image_id=f"sample_{sample_idx}",
            )
            cls_label = cfg.CLASS_NAMES[true_cls].replace(" ", "_")
            fname = save_dir / f"{cls_label}_sample{sample_idx}_gradcam.png"
            fig.savefig(fname, dpi=100, bbox_inches="tight")
            plt.close(fig)
            saved_paths.append(fname)

            sample_idx += 1

    cam.remove_hooks()

    # Grid figure
    grid_path = save_dir / "gradcam_grid_all_classes.png"
    fig_grid = create_gradcam_grid(
        originals_by_class, heatmaps_by_class,
        preds_by_class, confs_by_class,
        samples_per_class=num_samples_per_class,
        save_path=grid_path,
    )
    plt.close(fig_grid)
    saved_paths.append(grid_path)

    print(f"  Saved {len(saved_paths)} Grad-CAM figures to {save_dir}")
    return {
        "originals":  originals_by_class,
        "heatmaps":   heatmaps_by_class,
        "preds":      preds_by_class,
        "confs":      confs_by_class,
        "saved_paths": [str(p) for p in saved_paths],
    }


# ---------------------------------------------------------------------------
# 4. Misclassification analysis
# ---------------------------------------------------------------------------

def generate_gradcam_misclassified(
    model,
    val_loader:   DataLoader,
    device:       torch.device,
    num_samples:  int  = 10,
    save_dir:     Path = None,
    model_name:   str  = "dr_litenet_effb0",
) -> list[str]:
    """
    Find misclassified samples, generate Grad-CAM for both the predicted class
    and the true class, save side-by-side figures.
    Returns list of saved paths.
    """
    if save_dir is None:
        save_dir = cfg.GRAD_CAM_DIR / "misclassified"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    target_layer = model.get_cnn_last_conv_layer()
    cam          = GradCAM(model, target_layer)

    saved_paths = []
    count = 0
    sample_idx = 0

    for imgs, hc, labels in tqdm(val_loader, desc="  Grad-CAM (misclassified)", leave=False):
        if count >= num_samples:
            break

        for i in range(len(labels)):
            if count >= num_samples:
                break

            true_cls = int(labels[i].item())
            img_t    = imgs[i:i+1].to(device)
            hc_t     = hc[i:i+1].to(device)

            heatmap_pred, pred_cls, conf_pred = cam.generate(img_t, hc_t, target_class=None)

            if pred_cls == true_cls:
                sample_idx += 1
                continue   # skip correctly classified

            heatmap_true, _, conf_true = cam.generate(img_t, hc_t, target_class=true_cls)

            rgb = _tensor_to_rgb(imgs[i])

            # 2-row figure: top = predicted class CAM, bottom = true class CAM
            fig, axes = plt.subplots(2, 3, figsize=(12, 8))
            for row_idx, (hm, cls, conf, row_label) in enumerate([
                (heatmap_pred, pred_cls, conf_pred, "Predicted"),
                (heatmap_true, true_cls, conf_true, "True"),
            ]):
                overlay = overlay_gradcam(rgb, hm)
                axes[row_idx, 0].imshow(rgb);     axes[row_idx, 0].set_title("Original"); axes[row_idx, 0].axis("off")
                axes[row_idx, 1].imshow(hm, cmap="jet", vmin=0, vmax=1); axes[row_idx, 1].set_title(f"CAM ({row_label})"); axes[row_idx, 1].axis("off")
                axes[row_idx, 2].imshow(overlay); axes[row_idx, 2].set_title(f"{cfg.CLASS_NAMES[cls]} ({conf:.1%})"); axes[row_idx, 2].axis("off")

            pred_name = cfg.CLASS_NAMES[pred_cls]
            true_name = cfg.CLASS_NAMES[true_cls]
            fig.suptitle(f"Misclassified #{count+1}: Pred={pred_name}, True={true_name}", fontsize=11)
            plt.tight_layout()

            fname = save_dir / f"misclassified_{count+1:03d}_pred{pred_cls}_true{true_cls}.png"
            fig.savefig(fname, dpi=100, bbox_inches="tight")
            plt.close(fig)
            saved_paths.append(str(fname))
            count += 1
            sample_idx += 1

    cam.remove_hooks()
    print(f"  Saved {len(saved_paths)} misclassification Grad-CAM figures to {save_dir}")
    return saved_paths


# ---------------------------------------------------------------------------
# Sanity check — run as script
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import torch
    from src.dataset import get_dataloaders
    from src.model import build_model
    from src.utils import get_device, load_checkpoint

    print("=" * 60)
    print("explainability.py sanity check")
    print("=" * 60)

    device     = get_device()
    model_name = "dr_litenet_effb0"
    backbone   = "efficientnet_b0"

    # Load checkpoint
    ckpt_path = cfg.checkpoint_path(model_name, 1)
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

    model = build_model(backbone_name=backbone, pretrained=False, device=device)
    load_checkpoint(ckpt_path, model, device)
    model.eval()

    # Single-image Grad-CAM
    _, val_loader = get_dataloaders(fold_idx=0, batch_size=1, use_handcrafted=True, num_workers=0)
    imgs, hc, labels = next(iter(val_loader))

    target_layer = model.get_cnn_last_conv_layer()
    cam = GradCAM(model, target_layer)

    img_t = imgs[0:1].to(device)
    hc_t  = hc[0:1].to(device)
    true_cls = int(labels[0].item())

    heatmap, pred_cls, confidence = cam.generate(img_t, hc_t, target_class=None)

    print(f"\n  Input shape     : {tuple(img_t.shape)}")
    print(f"  Heatmap shape   : {heatmap.shape}   expected (224, 224)")
    print(f"  Heatmap range   : [{heatmap.min():.4f}, {heatmap.max():.4f}]   expected [0,1]")
    print(f"  Predicted class : {pred_cls} ({cfg.CLASS_NAMES[pred_cls]})")
    print(f"  True class      : {true_cls} ({cfg.CLASS_NAMES[true_cls]})")
    print(f"  Confidence      : {confidence:.4f}")

    assert heatmap.shape == (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE), "Heatmap shape mismatch"
    assert 0.0 <= heatmap.min() and heatmap.max() <= 1.0, "Heatmap values out of [0,1]"

    # Check hooks removed cleanly
    n_hooks_before = len(cam._hooks)
    cam.remove_hooks()
    n_hooks_after = len(cam._hooks)
    print(f"\n  Hooks before remove : {n_hooks_before}")
    print(f"  Hooks after  remove : {n_hooks_after}   expected 0")
    assert n_hooks_after == 0, "Hooks not removed!"

    # Save a single overlay figure
    rgb = _tensor_to_rgb(imgs[0])
    fig = create_gradcam_figure(
        rgb, heatmap, pred_cls, true_cls, confidence, image_id="sanity_sample"
    )
    cfg.GRAD_CAM_DIR.mkdir(parents=True, exist_ok=True)
    out_path = cfg.GRAD_CAM_DIR / "sanity_gradcam.png"
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [OK] Saved single figure: {out_path}")
    assert out_path.exists(), "Figure not saved"

    # Batch generation (3 samples per class from val loader)
    print("\n  Running batch Grad-CAM (correct classifications)...")
    _, val_loader_b = get_dataloaders(fold_idx=0, batch_size=4, use_handcrafted=True, num_workers=0)
    result = generate_gradcam_for_dataset(
        model=model, val_loader=val_loader_b, device=device,
        num_samples_per_class=2,
        save_dir=cfg.GRAD_CAM_DIR,
        model_name=model_name,
    )

    print(f"\n  Saved paths:")
    for p in result["saved_paths"]:
        print(f"    {p}")

    print("\n  Running misclassification Grad-CAM (5 samples)...")
    misc_paths = generate_gradcam_misclassified(
        model=model, val_loader=val_loader_b, device=device,
        num_samples=5, save_dir=cfg.GRAD_CAM_DIR / "misclassified",
    )
    for p in misc_paths:
        print(f"    {p}")

    # Summary
    print("\n--- All generated Grad-CAM figures ---")
    for f in sorted(cfg.GRAD_CAM_DIR.rglob("*.png")):
        print(f"  {f}")

    print("\n[ALL SANITY CHECKS PASSED]")
