"""
DR-LiteNet model architecture.

Components:
  CNNBranch            — pretrained lightweight backbone with GAP, head removed.
  ClassificationHead   — FC(d+52→512) → ReLU → Dropout → FC(512→5).
  DRLiteNet            — full dual-branch model with forward + extract_features.
  build_model()        — factory function (backbone selection, device placement).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torchvision.models as tvm

import src.config as cfg


# ---------------------------------------------------------------------------
# 1. CNN Backbone Branch
# ---------------------------------------------------------------------------

class CNNBranch(nn.Module):
    """
    Wraps a pretrained torchvision backbone (EfficientNetB0 or MobileNetV3-Small).
    Removes the classification head; keeps features + GAP.

    Output: (B, output_dim) flat deep feature vector.
    The last convolutional layer is accessible via `self.last_conv_layer`
    for Grad-CAM hook registration.
    """

    def __init__(self, backbone_name: str, pretrained: bool = True):
        super().__init__()

        weights_arg = "DEFAULT" if pretrained else None

        if backbone_name == "efficientnet_b0":
            base = tvm.efficientnet_b0(weights=weights_arg)
            self.features = base.features      # Conv → MBConv stack
            self.pool = base.avgpool           # AdaptiveAvgPool2d → (B,1280,1,1)
            self.output_dim = 1280

        elif backbone_name == "mobilenet_v3_small":
            base = tvm.mobilenet_v3_small(weights=weights_arg)
            self.features = base.features
            self.pool = base.avgpool           # AdaptiveAvgPool2d → (B,576,1,1)
            self.output_dim = 576

        else:
            raise ValueError(
                f"Unsupported backbone '{backbone_name}'. "
                "Choose 'efficientnet_b0' or 'mobilenet_v3_small'."
            )

        # Last conv layer = features[-1][0]  (Conv2dNormActivation → Conv2d)
        # Exposed for Grad-CAM hook registration.
        self.last_conv_layer: nn.Conv2d = self.features[-1][0]

        # Verify assumed output dim with a single dummy pass
        with torch.no_grad():
            dummy = torch.zeros(1, 3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)
            actual_dim = self._forward_features(dummy).shape[1]
        assert actual_dim == self.output_dim, (
            f"Backbone '{backbone_name}' GAP output dim mismatch: "
            f"expected {self.output_dim}, got {actual_dim}"
        )

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)        # (B, C, H, W)
        x = self.pool(x)            # (B, C, 1, 1)
        return torch.flatten(x, 1)  # (B, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_features(x)

    def freeze(self) -> None:
        """Freeze all backbone parameters (Phase 1 training)."""
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze all backbone parameters (Phase 2 fine-tuning)."""
        for p in self.parameters():
            p.requires_grad = True


# ---------------------------------------------------------------------------
# 2. Classification Head
# ---------------------------------------------------------------------------

class ClassificationHead(nn.Module):
    """
    Dense classification head:
      Linear(fused_dim → hidden_dim) → ReLU → Dropout → Linear(hidden_dim → num_classes)

    Takes the fused (deep + handcrafted) feature vector as input.
    Returns raw logits (softmax applied externally during loss computation / eval).
    """

    def __init__(
        self,
        fused_dim: int,
        hidden_dim: int = cfg.HIDDEN_DIM,
        num_classes: int = cfg.NUM_CLASSES,
        dropout_rate: float = cfg.DROPOUT_RATE,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# 3. DR-LiteNet — Full Dual-Branch Model
# ---------------------------------------------------------------------------

class DRLiteNet(nn.Module):
    """
    Dual-branch hybrid architecture for 5-class DR grading.

    Branch 1: CNN backbone (deep features, d-dim)
    Branch 2: Handcrafted lesion descriptors (52-dim, pre-extracted outside model)
    Fusion:   Concatenation → (d + 52)-dim
    Head:     ClassificationHead → 5 logits

    forward(images, handcrafted_features) → logits (B, 5)
    extract_features(images, handcrafted_features) → fused vector (B, d+52)
    """

    def __init__(
        self,
        backbone_name: str = cfg.BACKBONE,
        pretrained: bool = cfg.PRETRAINED,
        hidden_dim: int = cfg.HIDDEN_DIM,
        handcrafted_dim: int = cfg.HANDCRAFTED_FEATURE_DIM,
        num_classes: int = cfg.NUM_CLASSES,
        dropout_rate: float = cfg.DROPOUT_RATE,
    ):
        super().__init__()

        self.cnn_branch = CNNBranch(backbone_name, pretrained)
        self.backbone_name = backbone_name

        fused_dim = self.cnn_branch.output_dim + handcrafted_dim
        self.fused_dim = fused_dim

        self.classification_head = ClassificationHead(
            fused_dim=fused_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
        )

    def forward(
        self,
        images: torch.Tensor,
        handcrafted_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            images:               (B, 3, 224, 224)
            handcrafted_features: (B, 52)
        Returns:
            logits: (B, 5)
        """
        deep_features = self.cnn_branch(images)                          # (B, d)
        fused = torch.cat([deep_features, handcrafted_features], dim=1)  # (B, d+52)
        return self.classification_head(fused)                            # (B, 5)

    @torch.no_grad()
    def extract_features(
        self,
        images: torch.Tensor,
        handcrafted_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract fused feature vectors without computing logits.
        Used in Phase 1 training to collect features for SMOTE.

        Returns: (B, d+52) float tensor on the same device as the model.
        """
        self.eval()
        deep_features = self.cnn_branch(images)
        return torch.cat([deep_features, handcrafted_features], dim=1)

    def get_cnn_last_conv_layer(self) -> nn.Conv2d:
        """Return last Conv2d in CNN branch for Grad-CAM hook registration."""
        return self.cnn_branch.last_conv_layer

    def freeze_backbone(self) -> None:
        self.cnn_branch.freeze()

    def unfreeze_backbone(self) -> None:
        self.cnn_branch.unfreeze()

    def count_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# 4. Model Factory
# ---------------------------------------------------------------------------

def build_model(
    backbone_name: str = cfg.BACKBONE,
    pretrained: bool = cfg.PRETRAINED,
    device: torch.device = None,
) -> DRLiteNet:
    """
    Build and return a DRLiteNet instance.
    Moves the model to the specified device (defaults to cfg device detection).
    Prints parameter count and verifies < 10M constraint.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DRLiteNet(backbone_name=backbone_name, pretrained=pretrained)
    model.to(device)

    total_params = model.count_parameters(trainable_only=False)
    trainable_params = model.count_parameters(trainable_only=True)

    print(f"  Backbone        : {backbone_name}")
    print(f"  Fused dim       : {model.fused_dim}  ({model.cnn_branch.output_dim} CNN + 52 handcrafted)")
    print(f"  Total params    : {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    print(f"  Device          : {device}")

    assert total_params < 10_000_000, (
        f"Parameter count {total_params:,} exceeds 10M constraint (RQ3)."
    )
    print(f"  [OK] Parameter count < 10M constraint satisfied.")

    return model


# ---------------------------------------------------------------------------
# 5. Sanity check — run as script
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("DR-LiteNet Model Sanity Check")
    print("=" * 60)

    device = torch.device("cpu")

    for backbone_name in ["efficientnet_b0", "mobilenet_v3_small"]:
        print(f"\n--- Backbone: {backbone_name} ---")

        model = build_model(backbone_name=backbone_name, pretrained=False, device=device)
        model.eval()

        B = 2
        dummy_images = torch.randn(B, 3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)
        dummy_hc     = torch.randn(B, cfg.HANDCRAFTED_FEATURE_DIM)

        # Forward pass → logits
        logits = model(dummy_images, dummy_hc)
        print(f"  forward()        input : images{tuple(dummy_images.shape)}, hc{tuple(dummy_hc.shape)}")
        print(f"  forward()        output: {tuple(logits.shape)}   expected (2, 5)")
        assert logits.shape == (B, cfg.NUM_CLASSES), f"Logit shape mismatch: {logits.shape}"

        # extract_features → fused vector (for SMOTE)
        fused = model.extract_features(dummy_images, dummy_hc)
        expected_fused_dim = model.fused_dim
        print(f"  extract_features output: {tuple(fused.shape)}   expected (2, {expected_fused_dim})")
        assert fused.shape == (B, expected_fused_dim), f"Fused shape mismatch: {fused.shape}"

        # Last conv layer for Grad-CAM
        last_conv = model.get_cnn_last_conv_layer()
        print(f"  last_conv_layer : {type(last_conv).__name__}  "
              f"(out_channels={last_conv.out_channels})")
        assert isinstance(last_conv, torch.nn.Conv2d), "last_conv_layer is not a Conv2d"
        print(f"  [OK] All shape checks passed.")

    # Print architecture of default backbone
    print("\n" + "=" * 60)
    print("Full DRLiteNet architecture (EfficientNetB0, pretrained=False)")
    print("=" * 60)
    model_default = DRLiteNet(backbone_name="efficientnet_b0", pretrained=False)
    # Print head only (backbone is large)
    print("\n[ClassificationHead]")
    print(model_default.classification_head)
    print(f"\n[CNNBranch] backbone=efficientnet_b0  output_dim={model_default.cnn_branch.output_dim}")
    print(f"[Fused dim] {model_default.fused_dim}")

    print("\n[ALL SANITY CHECKS PASSED]")
