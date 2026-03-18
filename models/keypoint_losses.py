"""Losses and decoding utilities for heatmap-based keypoint training."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


class MaskedMSELoss(nn.Module):
    """Mean-squared error that ignores invisible keypoints."""

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        visibility: torch.Tensor,
    ) -> torch.Tensor:
        if predictions.shape != targets.shape:
            raise ValueError(
                f"Predictions and targets must have matching shapes, got {list(predictions.shape)} and {list(targets.shape)}."
            )
        visibility_mask = visibility[:, :, None, None].to(dtype=predictions.dtype)
        squared_error = (predictions - targets) ** 2
        masked_error = squared_error * visibility_mask
        denominator = torch.clamp(visibility_mask.sum() * predictions.shape[-1] * predictions.shape[-2], min=1.0)
        return masked_error.sum() / denominator


@dataclass(slots=True)
class DecodedHeatmaps:
    keypoints_xy: torch.Tensor
    confidence: torch.Tensor

    def to_dict(self) -> dict[str, Any]:
        return {
            "keypoints_xy": self.keypoints_xy,
            "confidence": self.confidence,
        }


def decode_heatmaps(heatmaps: torch.Tensor, *, image_size: int) -> DecodedHeatmaps:
    """Decode heatmaps by channel-wise argmax into image-space coordinates."""
    if heatmaps.ndim != 4:
        raise ValueError(f"Expected BxKxHxW heatmaps, got shape {list(heatmaps.shape)}")
    batch_size, num_keypoints, heatmap_height, heatmap_width = heatmaps.shape
    flattened = heatmaps.reshape(batch_size, num_keypoints, -1)
    confidence, indices = torch.max(flattened, dim=-1)
    xs = (indices % heatmap_width).to(dtype=torch.float32)
    ys = torch.div(indices, heatmap_width, rounding_mode="floor").to(dtype=torch.float32)
    scale_x = float(image_size) / float(heatmap_width)
    scale_y = float(image_size) / float(heatmap_height)
    keypoints_xy = torch.stack(((xs + 0.5) * scale_x, (ys + 0.5) * scale_y), dim=-1)
    return DecodedHeatmaps(keypoints_xy=keypoints_xy, confidence=confidence)
