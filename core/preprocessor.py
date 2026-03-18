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
    clahe_enabled: bool = False
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: tuple[int, int] = (8, 8)
    clahe_blend_alpha: float = 0.0


PREPROCESSING_PROFILES = ("default", "bone_window_v1")


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

        if self.config.clahe_enabled:
            clahe = cv2.createCLAHE(
                clipLimit=float(self.config.clahe_clip_limit),
                tileGridSize=tuple(int(value) for value in self.config.clahe_tile_grid_size),
            )
            clahe_input = np.clip(normalized * 255.0, 0.0, 255.0).astype(np.uint8)
            clahe_image = clahe.apply(clahe_input).astype(np.float32) / 255.0
            alpha = float(np.clip(self.config.clahe_blend_alpha, 0.0, 1.0))
            normalized = ((1.0 - alpha) * normalized) + (alpha * clahe_image)

        normalized = np.clip(normalized, 0.0, 1.0)
        return normalized.astype(np.float32)


def resolve_preprocessing_config(
    profile: str = "default",
    *,
    target_size: tuple[int, int] | None = None,
) -> PreprocessingConfig:
    """Build a preprocessing config from a named profile."""
    normalized_profile = str(profile or "default")
    if normalized_profile == "default":
        config = PreprocessingConfig()
    elif normalized_profile == "bone_window_v1":
        config = PreprocessingConfig(
            percentile_lower=0.5,
            percentile_upper=99.5,
            clahe_enabled=True,
            clahe_clip_limit=2.0,
            clahe_tile_grid_size=(8, 8),
            clahe_blend_alpha=0.7,
        )
    else:
        raise ValueError(
            f"Unsupported preprocessing profile '{normalized_profile}'. Expected one of: {', '.join(PREPROCESSING_PROFILES)}"
        )

    if target_size is not None:
        config.target_size = tuple(int(value) for value in target_size)
    return config


def get_preprocessor(
    config: PreprocessingConfig | None = None,
    *,
    profile: str = "default",
    target_size: tuple[int, int] | None = None,
) -> XRayPreprocessor:
    """Factory used by plugins and scripts."""
    resolved_config = config or resolve_preprocessing_config(profile, target_size=target_size)
    return XRayPreprocessor(config=resolved_config)
