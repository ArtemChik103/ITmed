"""
core/dicom_loader.py — DICOM file loader with metadata extraction.
"""
import logging
from typing import Tuple

import numpy as np
import pydicom

logger = logging.getLogger(__name__)


def load_dicom(file_path: str) -> Tuple[np.ndarray, dict]:
    """
    Загружает DICOM файл и возвращает нормализованный массив пикселей и метаданные.

    Args:
        file_path: Путь к .dcm файлу.

    Returns:
        Кортеж: (image_array float32, metadata_dict).
    """
    ds = pydicom.dcmread(file_path)

    # --- 1. Pixel Spacing ---
    if hasattr(ds, "PixelSpacing"):
        pixel_spacing: list[float] = [float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1])]
    elif hasattr(ds, "ImagerPixelSpacing"):
        pixel_spacing: list[float] = [float(ds.ImagerPixelSpacing[0]), float(ds.ImagerPixelSpacing[1])]
        logger.info("PixelSpacing не найден, используется ImagerPixelSpacing: %s", pixel_spacing)
    elif hasattr(ds, "NominalScannedPixelSpacing"):
        pixel_spacing: list[float] = [float(ds.NominalScannedPixelSpacing[0]), float(ds.NominalScannedPixelSpacing[1])]
        logger.info("PixelSpacing не найден, используется NominalScannedPixelSpacing: %s", pixel_spacing)
    elif hasattr(ds, "PixelAspectRatio"):
        pixel_spacing: list[float] = [float(ds.PixelAspectRatio[0]), float(ds.PixelAspectRatio[1])]
        logger.info("PixelSpacing не найден, используется PixelAspectRatio: %s", pixel_spacing)
    else:
        logger.warning("Атрибуты размера пикселя (PixelSpacing, ImagerPixelSpacing и др.) не найдены в файле '%s'. Используется default [1.0, 1.0].", file_path)
        pixel_spacing = [1.0, 1.0]

    # --- 2. Pixel array -> float32 ---
    pixel_array = ds.pixel_array.astype(np.float32)

    # --- 3. HU нормализация (Hounsfield Units) ---
    if hasattr(ds, "RescaleSlope") and hasattr(ds, "RescaleIntercept"):
        slope = float(ds.RescaleSlope)
        intercept = float(ds.RescaleIntercept)
        pixel_array = slope * pixel_array + intercept
        logger.debug("HU нормализация применена: slope=%.2f, intercept=%.2f", slope, intercept)

    # --- 4. Метаданные ---
    metadata: dict = {
        "pixel_spacing_mm": pixel_spacing,
        "patient_id": str(getattr(ds, "PatientID", "UNKNOWN")),
        "study_date": str(getattr(ds, "StudyDate", "UNKNOWN")),
        "modality": str(getattr(ds, "Modality", "UNKNOWN")),
        "image_shape": list(pixel_array.shape),
    }

    logger.info(
        "DICOM загружен: %s | shape=%s | modality=%s | pixel_spacing=%s",
        file_path,
        pixel_array.shape,
        metadata["modality"],
        pixel_spacing,
    )

    return pixel_array, metadata
