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


def apply_thin_plate_spline_warp(image: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Thin-Plate Spline (TPS) non-linear elastic morphing of anatomical structures."""
    try:
        h, w = image.shape[:2]
        grid_x, grid_y = np.meshgrid(
            np.linspace(w * 0.1, w * 0.9, 4),
            np.linspace(h * 0.1, h * 0.9, 4),
        )
        src = np.column_stack([grid_x.ravel(), grid_y.ravel()]).astype(np.float32)
        noise = np.random.uniform(-6.0, 6.0, src.shape).astype(np.float32)
        dst = src + noise

        matches = [cv2.DMatch(i, i, 0) for i in range(len(src))]
        tps = cv2.createThinPlateSplineShapeTransformer()
        tps.estimateTransformation(dst.reshape(1, -1, 2), src.reshape(1, -1, 2), matches)
        warped = tps.warpImage(image)
        return warped if warped is not None else image
    except Exception:
        return image


def _albumentations_base_pipeline(image_size: int) -> list[Any]:
    return [
        A.Resize(height=image_size, width=image_size, interpolation=cv2.INTER_AREA),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=1.0),
        ToTensorV2(),
    ]


def get_train_augmentations(image_size: int = 384) -> Any:
    """Return the training augmentation pipeline."""
    if _HAS_ALBUMENTATIONS:
        hole_size = max(8, int(image_size * 0.08))
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.03,
                    scale_limit=0.08,
                    rotate_limit=7,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.45,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.12,
                    contrast_limit=0.12,
                    p=0.35,
                ),
                A.CoarseDropout(
                    max_holes=6,
                    min_holes=1,
                    max_height=hole_size,
                    max_width=hole_size,
                    min_height=max(4, int(hole_size * 0.25)),
                    min_width=max(4, int(hole_size * 0.25)),
                    fill_value=0,
                    p=0.35,
                ),
                A.OneOf(
                    [
                        A.ElasticTransform(alpha=1.0, sigma=25, alpha_affine=15, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.4),
                        A.GridDistortion(num_steps=5, distort_limit=0.15, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.4),
                        A.Lambda(image=apply_thin_plate_spline_warp, p=0.4),
                    ],
                    p=0.35,
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


def apply_anatomical_hip_grafting(
    image_base: np.ndarray,
    image_donor: np.ndarray,
    hip_side: str = "left",
    blend_alpha: float = 0.5,
) -> np.ndarray:
    """Anatomical Hip Grafting / CutMix.

    Physiologically grafts a donor acetabular joint region onto a base pelvic radiograph
    with smooth cosine/Gaussian edge feathering to eliminate boundary artifacts.
    """
    if image_base.shape != image_donor.shape:
        image_donor = cv2.resize(image_donor, (image_base.shape[1], image_base.shape[0]))

    h, w = image_base.shape[:2]
    y1, y2 = int(h * 0.20), int(h * 0.80)
    if hip_side == "left":
        x1, x2 = int(w * 0.05), int(w * 0.48)
    else:
        x1, x2 = int(w * 0.52), int(w * 0.95)

    # Smooth feathered mask
    mask = np.zeros((h, w), dtype=np.float32)
    mask[y1:y2, x1:x2] = 1.0
    ksize = max(11, int(min(h, w) * 0.05) | 1)
    mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)

    if image_base.ndim == 3 and mask.ndim == 2:
        mask = mask[..., None]

    alpha_weight = blend_alpha * mask
    grafted = (1.0 - alpha_weight) * image_base.astype(np.float32) + alpha_weight * image_donor.astype(np.float32)
    return np.clip(grafted, 0, 255).astype(image_base.dtype)


def apply_native_16bit_contrast_jitter(
    image: np.ndarray,
    window_center_jitter: float = 0.15,
    window_width_jitter: float = 0.20,
    gamma_range: tuple[float, float] = (0.8, 1.25),
) -> np.ndarray:
    """Simulate radiologist window leveling on native high dynamic range radiographs.

    Jitters Window Center (WC) and Window Width (WW) to reveal trabecular bone
    versus unossified soft-tissue/cartilaginous interfaces.
    """
    img = image.astype(np.float32)
    min_val, max_val = float(img.min()), float(img.max())
    if max_val <= min_val:
        return image.copy()

    # Normalize to [0, 1]
    norm = (img - min_val) / (max_val - min_val)

    # Base center and width
    base_center = 0.5
    base_width = 0.8

    c_offset = float(np.random.uniform(-window_center_jitter, window_center_jitter))
    w_scale = float(np.random.uniform(1.0 - window_width_jitter, 1.0 + window_width_jitter))

    center = np.clip(base_center + c_offset, 0.15, 0.85)
    width = np.clip(base_width * w_scale, 0.2, 1.2)

    low = center - width / 2.0
    high = center + width / 2.0

    windowed = np.clip((norm - low) / (high - low), 0.0, 1.0)

    # Non-linear tissue gamma jitter
    gamma = float(np.random.uniform(gamma_range[0], gamma_range[1]))
    windowed = np.power(windowed, gamma)

    if np.issubdtype(image.dtype, np.integer):
        return (windowed * max_val).astype(image.dtype)
    return windowed.astype(image.dtype)


class Native16BitWindowJitter:
    """Albumentations-compatible augmentation for native DICOM high dynamic range jittering."""

    def __init__(
        self,
        p: float = 0.5,
        window_center_jitter: float = 0.15,
        window_width_jitter: float = 0.20,
    ) -> None:
        self.p = float(p)
        self.window_center_jitter = window_center_jitter
        self.window_width_jitter = window_width_jitter

    def __call__(self, image: np.ndarray, **kwargs: Any) -> dict[str, np.ndarray]:
        if np.random.rand() < self.p:
            image = apply_native_16bit_contrast_jitter(
                image,
                window_center_jitter=self.window_center_jitter,
                window_width_jitter=self.window_width_jitter,
            )
        return {"image": image}


