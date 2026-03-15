"""Lightweight preprocessing pipeline for X-ray images."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(slots=True)
class PreprocessingConfig:
    """Configurable preprocessing parameters."""

    target_size: tuple[int, int] = (512, 512)
    percentile_lower: float = 1.0
    percentile_upper: float = 99.0
    invert_monochrome1: bool = True


class XRayPreprocessor:
    """Normalize and resize X-ray images with predictable output."""

    def __init__(self, config: PreprocessingConfig | None = None) -> None:
        self.config = config or PreprocessingConfig()

    def preprocess(self, image: np.ndarray, metadata: dict[str, Any]) -> np.ndarray:
        if image.ndim == 3:
            if image.shape[0] == 1:
                image = image[0]
            elif image.shape[-1] == 1:
                image = image[..., 0]
            else:
                raise ValueError(f"Ожидается grayscale изображение, получена форма {list(image.shape)}")

        image = image.astype(np.float32, copy=False)

        if metadata.get("photometric_interpretation") == "MONOCHROME1" and self.config.invert_monochrome1:
            image = image.max() - image

        lower = float(np.percentile(image, self.config.percentile_lower))
        upper = float(np.percentile(image, self.config.percentile_upper))

        if upper <= lower:
            lower = float(image.min())
            upper = float(image.max())

        if upper <= lower:
            normalized = np.zeros_like(image, dtype=np.float32)
        else:
            normalized = np.clip(image, lower, upper)
            normalized = (normalized - lower) / (upper - lower)

        target_height, target_width = self.config.target_size
        if normalized.shape != (target_height, target_width):
            normalized = cv2.resize(
                normalized,
                (target_width, target_height),
                interpolation=cv2.INTER_AREA,
            )

        normalized = np.clip(normalized, 0.0, 1.0)
        return normalized.astype(np.float32)


def get_preprocessor(config: PreprocessingConfig | None = None) -> XRayPreprocessor:
    """Factory used by plugins and scripts."""
    return XRayPreprocessor(config=config)
