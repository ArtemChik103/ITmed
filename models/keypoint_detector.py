"""Heatmap-based keypoint detector with a ResNet encoder."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50

from models.classifier import load_backbone_weights


@dataclass(slots=True)
class KeypointDetectorConfig:
    architecture: str = "resnet50"
    num_keypoints: int = 8
    pretrained: bool = True
    pretrained_weights_path: str | None = None
    apply_sigmoid: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DeconvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class KeypointDetector(nn.Module):
    """ResNet-50 encoder with a simple deconvolution head for 8 heatmaps."""

    def __init__(
        self,
        *,
        num_keypoints: int = 8,
        pretrained: bool = True,
        pretrained_weights_path: str | Path | None = None,
        apply_sigmoid: bool = True,
    ) -> None:
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained and pretrained_weights_path is None else None
        self.backbone = resnet50(weights=weights)
        self.backbone.fc = nn.Identity()
        self.config = KeypointDetectorConfig(
            num_keypoints=int(num_keypoints),
            pretrained=bool(pretrained),
            pretrained_weights_path=str(pretrained_weights_path) if pretrained_weights_path else None,
            apply_sigmoid=bool(apply_sigmoid),
        )
        if pretrained and pretrained_weights_path is not None:
            load_backbone_weights(self.backbone, weights_path=pretrained_weights_path)

        self.head = nn.Sequential(
            DeconvBlock(2048, 256),
            DeconvBlock(256, 256),
            DeconvBlock(256, 256),
            nn.Conv2d(256, num_keypoints, kernel_size=1, stride=1, padding=0),
        )

    def freeze_encoder(self, frozen: bool = True) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = not frozen
        for parameter in self.head.parameters():
            parameter.requires_grad = True

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        heatmaps = self.head(x)
        if self.config.apply_sigmoid:
            heatmaps = torch.sigmoid(heatmaps)
        return heatmaps


def load_keypoint_detector_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[KeypointDetector, dict[str, Any]]:
    checkpoint = _torch_load(Path(checkpoint_path), map_location=device)
    model_config = checkpoint.get("model_config", {})
    model = KeypointDetector(
        num_keypoints=int(model_config.get("num_keypoints", 8)),
        pretrained=False,
        apply_sigmoid=bool(model_config.get("apply_sigmoid", True)),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint


def export_encoder_state_dict(model: KeypointDetector) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu()
        for key, value in model.backbone.state_dict().items()
        if not key.startswith("fc.")
    }


def _torch_load(path: Path, *, map_location: str | torch.device) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)
