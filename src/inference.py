"""
Inference script for DR-LiteNet.

Usage (CLI):
  # Single image or directory
  python src/inference.py --checkpoint experiments/dr_litenet_effb0/fold_1/checkpoints/best_model.pth \\
                          --input data/val_images/val_images/ \\
                          --gradcam

  # IDRiD external validation (when data is available)
  python src/inference.py --checkpoint <path> --idrid

Functions:
  load_model(checkpoint_path, backbone, device)
  predict_single(model, image_path, device, generate_gradcam, save_dir)
  run_inference(model, input_path, device, generate_gradcam, save_dir)
  evaluate_on_idrid(model, device)
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from typing import Union

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import src.config as cfg
from src.dataset import preprocess_image, extract_handcrafted_features
from src.explainability import GradCAM, overlay_gradcam, _tensor_to_rgb
from src.model import build_model
from src.utils import get_device, load_checkpoint, save_json


# ---------------------------------------------------------------------------
# 1. Model loading
# ---------------------------------------------------------------------------

def load_model(
    checkpoint_path: Union[str, Path],
    backbone:        str         = cfg.BACKBONE,
    device:          torch.device = None,
):
    """Load DRLiteNet from a saved checkpoint. Returns model in eval mode."""
    if device is None:
        device = get_device()
    model = build_model(backbone_name=backbone, pretrained=False, device=device)
    load_checkpoint(Path(checkpoint_path), model, device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# 2. Single-image prediction
# ---------------------------------------------------------------------------

def predict_single(
    model,
    image_path:      Union[str, Path],
    device:          torch.device,
    generate_gradcam: bool  = False,
    gradcam_save_dir: Path  = None,
) -> dict:
    """
    Preprocess one fundus image and predict DR severity.

    Returns dict matching STABLE §12:
    {
      "image":           "patient_001.jpg",
      "predicted_class": 2,
      "predicted_label": "Moderate",
      "probabilities":   [0.05, 0.10, 0.60, 0.15, 0.10],
      "grad_cam_path":   "outputs/grad_cam/..." or null
    }
    """
    image_path = Path(image_path)
    t0 = time.perf_counter()

    # --- Load & preprocess ---
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    rgb = preprocess_image(bgr)                          # (224, 224, 3) uint8

    # --- Handcrafted features ---
    hc_np   = extract_handcrafted_features(rgb)          # (52,) float32
    hc_t    = torch.from_numpy(hc_np).unsqueeze(0).to(device)  # (1, 52)

    # --- Image tensor ---
    from torchvision import transforms as T
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=cfg.IMAGENET_MEAN, std=cfg.IMAGENET_STD),
    ])
    from PIL import Image as PILImage
    img_t = transform(PILImage.fromarray(rgb)).unsqueeze(0).to(device)  # (1,3,224,224)

    # --- Forward pass ---
    with torch.no_grad():
        logits = model(img_t, hc_t)
        probs  = F.softmax(logits, dim=1)

    pred_class = int(logits.argmax(dim=1).item())
    probabilities = probs[0].cpu().numpy().tolist()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # --- Optional Grad-CAM ---
    grad_cam_path = None
    if generate_gradcam:
        if gradcam_save_dir is None:
            gradcam_save_dir = cfg.GRAD_CAM_DIR / "inference"
        gradcam_save_dir = Path(gradcam_save_dir)
        gradcam_save_dir.mkdir(parents=True, exist_ok=True)

        cam = GradCAM(model, model.get_cnn_last_conv_layer())
        heatmap, _, _ = cam.generate(img_t, hc_t, target_class=pred_class)
        cam.remove_hooks()

        # Overlay and save
        overlay = overlay_gradcam(rgb, heatmap)
        stem    = image_path.stem
        out_png = gradcam_save_dir / f"{stem}_gradcam.png"

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(rgb);                       axes[0].set_title("Original");      axes[0].axis("off")
        axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1); axes[1].set_title("Grad-CAM"); axes[1].axis("off")
        axes[2].imshow(overlay);                   axes[2].set_title(
            f"Pred: {cfg.CLASS_NAMES[pred_class]} ({probabilities[pred_class]:.1%})"
        ); axes[2].axis("off")
        fig.suptitle(f"{image_path.name}", fontsize=10)
        plt.tight_layout()
        fig.savefig(out_png, dpi=100, bbox_inches="tight")
        plt.close(fig)
        grad_cam_path = str(out_png)

    return {
        "image":           image_path.name,
        "predicted_class": pred_class,
        "predicted_label": cfg.CLASS_NAMES[pred_class],
        "probabilities":   [round(p, 6) for p in probabilities],
        "grad_cam_path":   grad_cam_path,
        "inference_ms":    round(elapsed_ms, 2),
    }


# ---------------------------------------------------------------------------
# 3. Batch inference
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def run_inference(
    model,
    input_path:       Union[str, Path],
    device:           torch.device,
    generate_gradcam: bool  = True,
    save_dir:         Path  = None,
    gradcam_save_dir: Path  = None,
    verbose:          bool  = True,
) -> list[dict]:
    """
    Run inference on a single image file or a directory of images.

    Returns list of prediction dicts. Saves:
      outputs/predictions/inference_results.json
      outputs/predictions/inference_results.csv
    """
    input_path = Path(input_path)

    # Collect image paths
    if input_path.is_file():
        image_paths = [input_path]
    elif input_path.is_dir():
        image_paths = sorted(
            p for p in input_path.iterdir()
            if p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if not image_paths:
        raise ValueError(f"No images found in {input_path}")

    if save_dir is None:
        save_dir = cfg.PREDICTIONS_DIR
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"  Running inference on {len(image_paths)} image(s)...")

    predictions = []
    for img_path in image_paths:
        try:
            result = predict_single(
                model, img_path, device,
                generate_gradcam=generate_gradcam,
                gradcam_save_dir=gradcam_save_dir,
            )
            predictions.append(result)
            if verbose:
                print(f"  {result['image']:40s}  "
                      f"class={result['predicted_class']} "
                      f"({result['predicted_label']})  "
                      f"conf={result['probabilities'][result['predicted_class']]:.3f}  "
                      f"{result['inference_ms']:.1f}ms")
        except Exception as e:
            print(f"  [WARN] Skipped {img_path.name}: {e}")

    # Save JSON
    json_path = save_dir / "inference_results.json"
    save_json(predictions, json_path)

    # Save CSV
    csv_path  = save_dir / "inference_results.csv"
    df_rows = []
    for r in predictions:
        row = {
            "image":           r["image"],
            "predicted_class": r["predicted_class"],
            "predicted_label": r["predicted_label"],
            "grad_cam_path":   r.get("grad_cam_path", ""),
            "inference_ms":    r.get("inference_ms", ""),
        }
        for c in range(cfg.NUM_CLASSES):
            row[f"prob_{c}"] = r["probabilities"][c]
        df_rows.append(row)
    pd.DataFrame(df_rows).to_csv(csv_path, index=False)

    if verbose:
        print(f"\n  Results saved to:")
        print(f"    JSON: {json_path}")
        print(f"    CSV : {csv_path}")

    return predictions


# ---------------------------------------------------------------------------
# 4. External validation on IDRiD
# ---------------------------------------------------------------------------

def evaluate_on_idrid(
    model,
    device:     torch.device,
    batch_size: int = cfg.BATCH_SIZE,
    num_workers: int = 2,
) -> dict:
    """
    Run full evaluation on IDRiD held-out set (when available).
    Identical preprocessing pipeline as APTOS.
    Saves: outputs/tables/external_validation.csv
           outputs/figures/confusion_matrix_idrid.png
    """
    if not cfg.IDRID_CSV.exists():
        print(f"  [SKIP] IDRiD labels not found at {cfg.IDRID_CSV}")
        print("  Download IDRiD and place at data/idrid/ to enable external validation.")
        return {}

    from src.dataset import APTOSDataset
    from src.evaluate import compute_all_metrics, plot_confusion_matrix
    from torch.utils.data import DataLoader
    import torch.nn.functional as F

    print("  Loading IDRiD dataset...")
    idrid_ds = APTOSDataset(
        image_dir=cfg.IDRID_IMAGE_DIR,
        csv_path=cfg.IDRID_CSV,
        augment=False,
        use_handcrafted=True,
    )
    idrid_loader = DataLoader(
        idrid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for imgs, hc, labels in idrid_loader:
            imgs, hc = imgs.to(device), hc.to(device)
            logits = model(imgs, hc)
            probs  = F.softmax(logits, dim=1)
            all_preds.extend(logits.argmax(1).cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
            all_probs.append(probs.cpu().numpy())

    y_probs = np.concatenate(all_probs, axis=0)
    metrics = compute_all_metrics(all_labels, all_preds, y_probs)

    print(f"  IDRiD — accuracy={metrics['accuracy']:.4f}  "
          f"QWK={metrics['qwk']:.4f}  macro_F1={metrics['macro_f1']:.4f}")

    # Save confusion matrix figure
    import numpy as np
    cm_path = plot_confusion_matrix(
        np.array(metrics["confusion_matrix"]),
        averaged=False, fold_idx=0, model_name="idrid",
    )
    # Rename to idrid-specific filename
    idrid_cm_path = cfg.FIGURES_DIR / "confusion_matrix_idrid.png"
    cm_path.rename(idrid_cm_path)

    # Save external validation table
    cfg.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    ext_val_path = cfg.TABLES_DIR / "external_validation.csv"
    rows = [
        {"metric": "accuracy",  "value": round(metrics["accuracy"],  6)},
        {"metric": "qwk",       "value": round(metrics["qwk"],       6)},
        {"metric": "macro_f1",  "value": round(metrics["macro_f1"],  6)},
    ]
    for c in range(cfg.NUM_CLASSES):
        rows.append({"metric": f"sens_{c}_{cfg.CLASS_NAMES[c]}", "value": round(metrics["per_class_sensitivity"][c], 6)})
        rows.append({"metric": f"auc_{c}_{cfg.CLASS_NAMES[c]}",  "value": round(metrics["per_class_auc"][c], 6)})
    pd.DataFrame(rows).to_csv(ext_val_path, index=False)
    print(f"  Saved external validation table: {ext_val_path}")

    return metrics


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    ap = argparse.ArgumentParser(description="DR-LiteNet inference")
    ap.add_argument("--checkpoint",  type=str, required=True,
                    help="Path to best_model.pth checkpoint")
    ap.add_argument("--backbone",    type=str, default=cfg.BACKBONE,
                    help="Backbone name (default from config)")
    ap.add_argument("--input",       type=str, default=None,
                    help="Path to image file or directory")
    ap.add_argument("--output",      type=str, default=str(cfg.PREDICTIONS_DIR),
                    help="Directory to save predictions")
    ap.add_argument("--gradcam",     action="store_true",
                    help="Generate Grad-CAM overlays alongside predictions")
    ap.add_argument("--idrid",       action="store_true",
                    help="Run IDRiD external validation instead of inference")
    ap.add_argument("--no_gradcam",  action="store_true",
                    help="Disable Grad-CAM even if --gradcam not set")
    args = ap.parse_args()

    device = get_device()
    model  = load_model(args.checkpoint, backbone=args.backbone, device=device)

    if args.idrid:
        evaluate_on_idrid(model, device)
    elif args.input:
        run_inference(
            model=model,
            input_path=Path(args.input),
            device=device,
            generate_gradcam=args.gradcam and not args.no_gradcam,
            save_dir=Path(args.output),
        )
    else:
        print("Provide --input <image_or_dir> or --idrid.")
        ap.print_help()


# ---------------------------------------------------------------------------
# Sanity check — run as script
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys as _sys
    # If no CLI args, run the built-in sanity check
    if len(_sys.argv) == 1:
        print("=" * 60)
        print("inference.py sanity check")
        print("=" * 60)

        device = get_device()
        ckpt_path = cfg.checkpoint_path("dr_litenet_effb0", 1)
        assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

        model = load_model(ckpt_path, backbone="efficientnet_b0", device=device)
        print(f"  Model loaded from: {ckpt_path}")

        # --- Single image prediction ---
        sample_images = sorted(cfg.APTOS_VAL_DIR.iterdir())[:3]
        assert sample_images, "No val images found"

        print(f"\n  Running predict_single on: {sample_images[0].name}")
        result = predict_single(
            model, sample_images[0], device,
            generate_gradcam=True,
            gradcam_save_dir=cfg.GRAD_CAM_DIR / "inference",
        )

        print("\n  --- Prediction output (STABLE §12 format) ---")
        import json as _json
        print(_json.dumps(result, indent=4))

        # Verify required keys
        required = ["image", "predicted_class", "predicted_label", "probabilities", "grad_cam_path"]
        for k in required:
            assert k in result, f"Missing key: {k}"
        assert len(result["probabilities"]) == cfg.NUM_CLASSES, "Wrong prob vector length"
        assert abs(sum(result["probabilities"]) - 1.0) < 1e-4, "Probabilities don't sum to 1"
        assert result["grad_cam_path"] is not None, "Grad-CAM path is None"
        assert Path(result["grad_cam_path"]).exists(), "Grad-CAM file not saved"
        print(f"\n  [OK] All STABLE §12 keys present")
        print(f"  [OK] Probabilities sum to 1.0")
        print(f"  [OK] Grad-CAM saved: {result['grad_cam_path']}")

        # --- Batch inference (3 images) ---
        print(f"\n  Running batch inference on {len(sample_images)} images...")
        predictions = run_inference(
            model=model,
            input_path=cfg.APTOS_VAL_DIR,
            device=device,
            generate_gradcam=False,   # skip gradcam for speed in batch sanity
            save_dir=cfg.PREDICTIONS_DIR,
            verbose=True,
        )
        # Limit output to first 3
        predictions = predictions[:3]

        # Verify JSON saved
        json_path = cfg.PREDICTIONS_DIR / "inference_results.json"
        csv_path  = cfg.PREDICTIONS_DIR / "inference_results.csv"
        assert json_path.exists(), "inference_results.json not saved"
        assert csv_path.exists(),  "inference_results.csv not saved"
        print(f"\n  [OK] inference_results.json saved ({json_path.stat().st_size} bytes)")
        print(f"  [OK] inference_results.csv  saved ({csv_path.stat().st_size} bytes)")

        # Measure inference speed (10 runs)
        t0 = time.perf_counter()
        for _ in range(10):
            predict_single(model, sample_images[0], device, generate_gradcam=False)
        avg_ms = (time.perf_counter() - t0) / 10 * 1000
        print(f"\n  Inference speed: {avg_ms:.1f} ms/image (avg of 10 runs, CPU)")

        # IDRiD check
        print(f"\n  IDRiD available: {cfg.IDRID_CSV.exists()}")
        if not cfg.IDRID_CSV.exists():
            print("  [SKIP] IDRiD not present — external validation skipped (expected).")

        # List all saved inference outputs
        print("\n  --- Saved inference outputs ---")
        for p in sorted(cfg.PREDICTIONS_DIR.glob("*")):
            print(f"    {p.name}  ({p.stat().st_size:,} bytes)")
        for p in sorted((cfg.GRAD_CAM_DIR / "inference").glob("*.png")):
            print(f"    grad_cam/inference/{p.name}")

        print("\n[ALL SANITY CHECKS PASSED]")
    else:
        main()
