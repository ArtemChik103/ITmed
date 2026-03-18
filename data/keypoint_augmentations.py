"""Keypoint-aware augmentation helpers for MTDDH anatomy pretraining."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import torch
from torchvision.transforms import functional as TF

from data.augmentations import IMAGENET_MEAN, IMAGENET_STD

LEFT_RIGHT_KEYPOINT_PAIRS = ((0, 4), (1, 5), (2, 6), (3, 7))


@dataclass(slots=True)
class KeypointAugmentationConfig:
    target_size: int = 384
    train: bool = False
    rotation_limit: float = 5.0
    translate_limit: float = 0.02
    scale_limit: float = 0.03
    brightness_limit: float = 0.08
    contrast_limit: float = 0.08
    flip_enabled: bool = False
    flip_probability: float = 0.0


def _normalize_image(image: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(np.transpose(image, (2, 0, 1)))).to(dtype=torch.float32)
    return TF.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)


def _apply_horizontal_flip(
    image: np.ndarray,
    keypoints_xy: np.ndarray,
    visibility: np.ndarray,
    bbox: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    flipped = np.ascontiguousarray(image[:, ::-1])
    width = float(image.shape[1])
    keypoints_xy = keypoints_xy.copy()
    visibility = visibility.copy()
    bbox = bbox.copy()

    visible_indices = visibility > 0
    keypoints_xy[visible_indices, 0] = (width - 1.0) - keypoints_xy[visible_indices, 0]
    for left_index, right_index in LEFT_RIGHT_KEYPOINT_PAIRS:
        keypoints_xy[[left_index, right_index]] = keypoints_xy[[right_index, left_index]]
        visibility[[left_index, right_index]] = visibility[[right_index, left_index]]

    bbox[0] = (width - 1.0) - (bbox[0] + bbox[2])
    return flipped, keypoints_xy, visibility, bbox


class KeypointTransform:
    """Lightweight numpy-based transform that preserves keypoint geometry."""

    def __init__(self, config: KeypointAugmentationConfig | None = None) -> None:
        self.config = config or KeypointAugmentationConfig()

    def __call__(
        self,
        *,
        image: np.ndarray,
        keypoints_xy: np.ndarray,
        visibility: np.ndarray,
        bbox: np.ndarray,
    ) -> dict[str, Any]:
        config = self.config
        target_size = int(config.target_size)
        original_height, original_width = image.shape[:2]
        scale_x = target_size / float(max(original_width, 1))
        scale_y = target_size / float(max(original_height, 1))

        resized = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_AREA)
        keypoints_xy = keypoints_xy.astype(np.float32, copy=True)
        visibility = visibility.astype(np.float32, copy=True)
        bbox = bbox.astype(np.float32, copy=True)

        visible_indices = visibility > 0
        keypoints_xy[visible_indices, 0] *= scale_x
        keypoints_xy[visible_indices, 1] *= scale_y
        bbox[0] *= scale_x
        bbox[1] *= scale_y
        bbox[2] *= scale_x
        bbox[3] *= scale_y

        if config.train:
            if config.flip_enabled and random.random() < float(config.flip_probability):
                resized, keypoints_xy, visibility, bbox = _apply_horizontal_flip(
                    resized,
                    keypoints_xy,
                    visibility,
                    bbox,
                )

            center = (target_size / 2.0, target_size / 2.0)
            angle = random.uniform(-config.rotation_limit, config.rotation_limit)
            scale = 1.0 + random.uniform(-config.scale_limit, config.scale_limit)
            translate_x = random.uniform(-config.translate_limit, config.translate_limit) * target_size
            translate_y = random.uniform(-config.translate_limit, config.translate_limit) * target_size
            matrix = cv2.getRotationMatrix2D(center, angle, scale)
            matrix[0, 2] += translate_x
            matrix[1, 2] += translate_y

            resized = cv2.warpAffine(
                resized,
                matrix,
                (target_size, target_size),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )

            homogeneous = np.concatenate(
                [keypoints_xy[:, :2], np.ones((keypoints_xy.shape[0], 1), dtype=np.float32)],
                axis=1,
            )
            transformed = homogeneous @ matrix.T
            keypoints_xy[visible_indices] = transformed[visible_indices]
            outside = (
                (keypoints_xy[:, 0] < 0.0)
                | (keypoints_xy[:, 0] > (target_size - 1.0))
                | (keypoints_xy[:, 1] < 0.0)
                | (keypoints_xy[:, 1] > (target_size - 1.0))
            )
            visibility[outside] = 0.0

            bbox_corners = np.array(
                [
                    [bbox[0], bbox[1], 1.0],
                    [bbox[0] + bbox[2], bbox[1], 1.0],
                    [bbox[0], bbox[1] + bbox[3], 1.0],
                    [bbox[0] + bbox[2], bbox[1] + bbox[3], 1.0],
                ],
                dtype=np.float32,
            )
            transformed_corners = bbox_corners @ matrix.T
            min_x = float(np.clip(np.min(transformed_corners[:, 0]), 0.0, target_size - 1.0))
            min_y = float(np.clip(np.min(transformed_corners[:, 1]), 0.0, target_size - 1.0))
            max_x = float(np.clip(np.max(transformed_corners[:, 0]), 0.0, target_size - 1.0))
            max_y = float(np.clip(np.max(transformed_corners[:, 1]), 0.0, target_size - 1.0))
            bbox = np.array([min_x, min_y, max(max_x - min_x, 1.0), max(max_y - min_y, 1.0)], dtype=np.float32)

            brightness_factor = 1.0 + random.uniform(-config.brightness_limit, config.brightness_limit)
            contrast_factor = 1.0 + random.uniform(-config.contrast_limit, config.contrast_limit)
            resized = np.clip((resized * contrast_factor) + (brightness_factor - 1.0), 0.0, 1.0)

        return {
            "image": _normalize_image(resized.astype(np.float32, copy=False)),
            "keypoints_xy": keypoints_xy.astype(np.float32, copy=False),
            "visibility": visibility.astype(np.float32, copy=False),
            "bbox": bbox.astype(np.float32, copy=False),
        }


def get_keypoint_train_augmentations(
    *,
    target_size: int = 384,
    flip_enabled: bool = False,
) -> KeypointTransform:
    return KeypointTransform(
        KeypointAugmentationConfig(
            target_size=target_size,
            train=True,
            flip_enabled=flip_enabled,
            flip_probability=0.5 if flip_enabled else 0.0,
        )
    )


def get_keypoint_eval_augmentations(*, target_size: int = 384) -> KeypointTransform:
    return KeypointTransform(KeypointAugmentationConfig(target_size=target_size, train=False))
