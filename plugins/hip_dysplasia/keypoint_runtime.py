"""Optional runtime for MTDDH keypoint checkpoints used in education mode."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from data.import_mtddh_keypoints import RAW_KEYPOINT_NAMES
from models.keypoint_detector import load_keypoint_detector_from_checkpoint
from models.keypoint_losses import decode_heatmaps
from plugins.hip_dysplasia.model import IMAGENET_MEAN, IMAGENET_STD, REPO_ROOT


@dataclass(slots=True)
class KeypointPrediction:
    """Decoded keypoint prediction in the coordinate space of the input image."""

    keypoints_xy: list[tuple[float, float]]
    keypoint_names: list[str]
    input_size: int
    model_loaded: bool
    score_map: np.ndarray | None = None


def resolve_keypoint_checkpoint_path(explicit_path: str | Path | None = None) -> Path | None:
    """Resolve the optional keypoint checkpoint path from an explicit value or env var."""
    candidate = explicit_path or os.getenv("HIP_DYSPLASIA_KEYPOINT_CHECKPOINT")
    if not candidate:
        return None

    path = Path(candidate)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path if path.exists() else None


def resolve_keypoint_device(device_name: str | None = None) -> torch.device:
    """Resolve the inference device for the optional keypoint runtime."""
    resolved_name = device_name or os.getenv("HIP_DYSPLASIA_KEYPOINT_DEVICE") or "auto"
    if resolved_name != "auto":
        if resolved_name.startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(resolved_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def prepare_keypoint_image_tensor(image: np.ndarray, *, input_size: int) -> torch.Tensor:
    """Convert a normalized grayscale image into the tensor format expected by the keypoint model."""
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D image for keypoint inference, got shape {list(image.shape)}")

    image = np.clip(image.astype(np.float32, copy=False), 0.0, 1.0)
    if image.shape != (input_size, input_size):
        image = cv2.resize(image, (input_size, input_size), interpolation=cv2.INTER_AREA)

    rgb_image = np.repeat(image[..., None], 3, axis=2)
    normalized = (rgb_image - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(np.transpose(normalized, (2, 0, 1))).to(dtype=torch.float32)
    return tensor.unsqueeze(0)


class HipDysplasiaKeypointRuntime:
    """Standalone keypoint runtime that can be enabled without changing classifier inference."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        *,
        device: str | torch.device | None = None,
    ) -> None:
        resolved_checkpoint = resolve_keypoint_checkpoint_path(checkpoint_path)
        if resolved_checkpoint is None:
            raise FileNotFoundError(
                "Keypoint checkpoint not found. Set HIP_DYSPLASIA_KEYPOINT_CHECKPOINT to enable the anatomy overlay."
            )

        resolved_device = resolve_keypoint_device(str(device) if device is not None else None)
        self.checkpoint_path = resolved_checkpoint
        self.device = resolved_device
        self.model, self.checkpoint = load_keypoint_detector_from_checkpoint(
            self.checkpoint_path,
            device=self.device,
        )
        training_config = self.checkpoint.get("training_config", {})
        self.input_size = int(training_config.get("input_size", 384))
        self.keypoint_names = list(RAW_KEYPOINT_NAMES)
        self.model_loaded = True

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> KeypointPrediction:
        if image.ndim == 3 and image.shape[-1] == 1:
            image = image[..., 0]
        if image.ndim != 2:
            raise ValueError(f"Expected a 2D image for keypoint inference, got shape {list(image.shape)}")

        source_height, source_width = int(image.shape[0]), int(image.shape[1])
        tensor = prepare_keypoint_image_tensor(image, input_size=self.input_size).to(self.device)
        heatmaps = self.model(tensor)
        decoded = decode_heatmaps(heatmaps, image_size=self.input_size)

        coordinates = decoded.keypoints_xy[0].detach().cpu().numpy().astype(np.float32, copy=False)
        scale_x = source_width / float(max(self.input_size, 1))
        scale_y = source_height / float(max(self.input_size, 1))
        coordinates[:, 0] = np.clip(coordinates[:, 0] * scale_x, 0.0, max(source_width - 1.0, 0.0))
        coordinates[:, 1] = np.clip(coordinates[:, 1] * scale_y, 0.0, max(source_height - 1.0, 0.0))

        return KeypointPrediction(
            keypoints_xy=[(float(x), float(y)) for x, y in coordinates.tolist()],
            keypoint_names=list(self.keypoint_names),
            input_size=self.input_size,
            model_loaded=self.model_loaded,
            score_map=heatmaps[0].detach().cpu().numpy(),
        )
