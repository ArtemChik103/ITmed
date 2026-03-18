"""Classifier models for Phase 3 hip dysplasia experiments."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torchvision.models import (
    ResNet34_Weights,
    ResNet50_Weights,
    resnet34,
    resnet50,
)

ARCHITECTURES: dict[str, dict[str, Any]] = {
    "resnet34": {
        "builder": resnet34,
        "weights_enum": ResNet34_Weights,
        "default_weights": "IMAGENET1K_V1",
        "feature_dim": 512,
    },
    "resnet50": {
        "builder": resnet50,
        "weights_enum": ResNet50_Weights,
        "default_weights": "IMAGENET1K_V2",
        "feature_dim": 2048,
    },
}


@dataclass(slots=True)
class ClassifierConfig:
    """Serializable classifier configuration."""

    architecture: str = "resnet50"
    dropout: float = 0.3
    pretrained: bool = True
    pretrained_weights_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_pretrained_weights(
    model: nn.Module,
    *,
    weights_path: str | Path,
) -> None:
    load_backbone_weights(model, weights_path=weights_path)


def _extract_backbone_state_dict(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Expected a dictionary-like checkpoint payload for pretrained weights.")

    if "encoder_state_dict" in payload and isinstance(payload["encoder_state_dict"], dict):
        return dict(payload["encoder_state_dict"])
    if "state_dict" in payload and isinstance(payload["state_dict"], dict):
        return _extract_backbone_state_dict(payload["state_dict"])
    if "model_state" in payload and isinstance(payload["model_state"], dict):
        return _extract_backbone_state_dict(payload["model_state"])

    return dict(payload)


def load_backbone_weights(
    model: nn.Module,
    *,
    weights_path: str | Path,
) -> None:
    state_dict = _extract_backbone_state_dict(_torch_load(Path(weights_path), map_location="cpu"))

    filtered_state_dict: dict[str, Any] = {}
    for key, value in state_dict.items():
        normalized_key = str(key)
        if normalized_key.startswith("backbone."):
            normalized_key = normalized_key.removeprefix("backbone.")
        if normalized_key.startswith("encoder."):
            normalized_key = normalized_key.removeprefix("encoder.")
        if normalized_key.startswith("module."):
            normalized_key = normalized_key.removeprefix("module.")
        if normalized_key.startswith("fc.") or normalized_key.startswith("head.") or normalized_key.startswith("classifier."):
            continue
        filtered_state_dict[normalized_key] = value

    model.load_state_dict(filtered_state_dict, strict=False)


class HipDysplasiaClassifier(nn.Module):
    """Binary classifier with a configurable ResNet backbone."""

    def __init__(
        self,
        *,
        architecture: str = "resnet50",
        dropout: float = 0.3,
        pretrained: bool = True,
        pretrained_weights_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        if architecture not in ARCHITECTURES:
            supported = ", ".join(sorted(ARCHITECTURES))
            raise ValueError(f"Unsupported architecture '{architecture}'. Expected one of: {supported}")

        self.config = ClassifierConfig(
            architecture=architecture,
            dropout=dropout,
            pretrained=pretrained,
            pretrained_weights_path=str(pretrained_weights_path) if pretrained_weights_path else None,
        )

        architecture_spec = ARCHITECTURES[architecture]
        weights = None
        if pretrained and pretrained_weights_path is None:
            weights_enum = architecture_spec["weights_enum"]
            weights = getattr(weights_enum, architecture_spec["default_weights"])

        builder = architecture_spec["builder"]
        self.backbone = builder(weights=weights)
        feature_dim = int(architecture_spec["feature_dim"])
        self.backbone.fc = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 1),
        )

        if pretrained and pretrained_weights_path is not None:
            _load_pretrained_weights(self.backbone, weights_path=pretrained_weights_path)

    def freeze_backbone(self, frozen: bool = True) -> None:
        """Freeze or unfreeze the convolutional backbone, keeping the head trainable."""
        for name, parameter in self.backbone.named_parameters():
            parameter.requires_grad = not frozen or name.startswith("fc.")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = self.backbone(inputs)
        return logits.reshape(-1)


def load_classifier_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[HipDysplasiaClassifier, dict[str, Any]]:
    """Instantiate a classifier and load its checkpoint state."""
    checkpoint = _torch_load(Path(checkpoint_path), map_location=device)
    config_data = checkpoint.get("model_config", {})
    model = HipDysplasiaClassifier(
        architecture=config_data.get("architecture", "resnet50"),
        dropout=float(config_data.get("dropout", 0.3)),
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint


def _torch_load(path: Path, *, map_location: str | torch.device) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)
