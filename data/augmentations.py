"""Augmentation helpers for Phase 3 classifier training.

Albumentations is preferred when available. Kaggle offline notebooks may not
have it installed, so this module also provides a torchvision-based fallback.
"""
from __future__ import annotations

import random
from typing import Any

import cv2
import numpy as np
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    _HAS_ALBUMENTATIONS = True
except ImportError:
    A = None
    ToTensorV2 = None
    _HAS_ALBUMENTATIONS = False

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _to_tensor(image: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(image, torch.Tensor):
        tensor = image
    else:
        tensor = torch.from_numpy(np.ascontiguousarray(np.transpose(image, (2, 0, 1))))
    return tensor.to(dtype=torch.float32)


def _resize_and_normalize(image: np.ndarray | torch.Tensor, *, image_size: int) -> torch.Tensor:
    tensor = _to_tensor(image)
    tensor = TF.resize(
        tensor,
        [image_size, image_size],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    return TF.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)


class _TorchvisionFallbackCompose:
    """Minimal Compose-like wrapper that matches the dataset call contract."""

    def __init__(self, *, image_size: int, train: bool) -> None:
        self._image_size = int(image_size)
        self._train = bool(train)

    def __call__(self, *, image: np.ndarray | torch.Tensor, **_: Any) -> dict[str, torch.Tensor]:
        tensor = _to_tensor(image)
        if self._train:
            if random.random() < 0.5:
                tensor = TF.hflip(tensor)

            height, width = tensor.shape[-2:]
            translate = [
                int(round(random.uniform(-0.02, 0.02) * width)),
                int(round(random.uniform(-0.02, 0.02) * height)),
            ]
            scale = max(0.92, min(1.08, 1.0 + random.uniform(-0.08, 0.08)))
            angle = random.uniform(-10.0, 10.0)
            tensor = TF.affine(
                tensor,
                angle=angle,
                translate=translate,
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.BILINEAR,
                fill=0.0,
            )

            if random.random() < 0.25:
                brightness_factor = 1.0 + random.uniform(-0.1, 0.1)
                contrast_factor = 1.0 + random.uniform(-0.1, 0.1)
                tensor = TF.adjust_brightness(tensor, brightness_factor)
                tensor = TF.adjust_contrast(tensor, contrast_factor)
                tensor = torch.clamp(tensor, min=0.0, max=1.0)

        tensor = _resize_and_normalize(tensor, image_size=self._image_size)
        return {"image": tensor}


def _albumentations_base_pipeline(image_size: int) -> list[Any]:
    return [
        A.Resize(height=image_size, width=image_size, interpolation=cv2.INTER_AREA),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=1.0),
        ToTensorV2(),
    ]


def get_train_augmentations(image_size: int = 384) -> Any:
    """Return the training augmentation pipeline."""
    if _HAS_ALBUMENTATIONS:
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.02,
                    scale_limit=0.08,
                    rotate_limit=10,
                    border_mode=cv2.BORDER_REPLICATE,
                    p=0.4,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1,
                    contrast_limit=0.1,
                    p=0.25,
                ),
                *_albumentations_base_pipeline(image_size),
            ]
        )
    return _TorchvisionFallbackCompose(image_size=image_size, train=True)


def get_eval_augmentations(image_size: int = 384) -> Any:
    """Return the deterministic validation/inference pipeline."""
    if _HAS_ALBUMENTATIONS:
        return A.Compose(_albumentations_base_pipeline(image_size))
    return _TorchvisionFallbackCompose(image_size=image_size, train=False)
