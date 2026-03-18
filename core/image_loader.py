"""Unified image loading for DICOM and raster X-ray inputs."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from core.dicom_loader import load_dicom

RASTER_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
DICOM_EXTENSIONS = {".dcm"}
SUPPORTED_MEDICAL_IMAGE_EXTENSIONS = DICOM_EXTENSIONS | RASTER_EXTENSIONS


def _normalize_raster_shape(array: np.ndarray) -> tuple[np.ndarray, int, bool]:
    """Collapse raster data into a single grayscale plane."""
    if array.ndim == 2:
        return array.astype(np.float32, copy=False), 1, True

    if array.ndim == 3:
        if array.shape[2] == 1:
            return array[..., 0].astype(np.float32, copy=False), 1, True

        visible_channels = array[..., :3].astype(np.float32, copy=False)
        grayscale = np.mean(visible_channels, axis=2)
        return grayscale.astype(np.float32, copy=False), int(array.shape[2]), False

    raise ValueError(f"Unsupported raster image shape: {list(array.shape)}")


def _read_raster_array(path: Path) -> np.ndarray:
    """Read a raster image with OpenCV first and PIL as a fallback."""
    cv2_array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if cv2_array is not None:
        return cv2_array

    with Image.open(path) as image:
        return np.asarray(image)


def _load_raster_image(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a PNG/JPEG/BMP image and expose DICOM-like metadata keys."""
    raw_array = _read_raster_array(path)
    grayscale_array, channel_count, is_grayscale = _normalize_raster_shape(raw_array)
    bit_depth = int(raw_array.dtype.itemsize * 8)

    metadata: dict[str, Any] = {
        "pixel_spacing_mm": [],
        "pixel_spacing_source": None,
        "imager_pixel_spacing_mm": None,
        "patient_id": None,
        "study_instance_uid": None,
        "study_date": None,
        "study_date_source": None,
        "modality": None,
        "sop_class_uid": None,
        "photometric_interpretation": None,
        "samples_per_pixel": channel_count,
        "number_of_frames": 1,
        "bits_allocated": bit_depth,
        "image_shape": list(grayscale_array.shape),
        "source_format": path.suffix.lower().lstrip("."),
        "channels": channel_count,
        "is_grayscale": bool(is_grayscale),
    }
    return grayscale_array.astype(np.float32, copy=False), metadata


def load_medical_image(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a DICOM or raster image and return grayscale pixels plus metadata."""
    resolved_path = Path(path)
    suffix = resolved_path.suffix.lower()

    if suffix in DICOM_EXTENSIONS:
        image, metadata = load_dicom(str(resolved_path))
        payload = dict(metadata)
        payload.setdefault("source_format", "dcm")
        return image.astype(np.float32, copy=False), payload

    if suffix in RASTER_EXTENSIONS:
        return _load_raster_image(resolved_path)

    raise ValueError(
        f"Unsupported medical image format '{resolved_path.suffix}'. "
        f"Supported extensions: {sorted(SUPPORTED_MEDICAL_IMAGE_EXTENSIONS)}"
    )
