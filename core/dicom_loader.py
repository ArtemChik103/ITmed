"""DICOM loader with metadata extraction tailored for X-ray images."""
from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import pydicom
from pydicom.dataset import Dataset

logger = logging.getLogger(__name__)

DATE_FIELDS = (
    "StudyDate",
    "SeriesDate",
    "AcquisitionDate",
    "ContentDate",
    "InstanceCreationDate",
)


def _extract_text(ds: Dataset, field_name: str) -> str | None:
    value = getattr(ds, field_name, None)
    if value is None:
        return None

    text = str(value).strip()
    return text or None


def _extract_float_pair(value: Any) -> list[float] | None:
    if value is None:
        return None

    try:
        values = list(value)
    except TypeError:
        return None

    if len(values) < 2:
        return None

    try:
        return [float(values[0]), float(values[1])]
    except (TypeError, ValueError):
        return None


def _extract_int(value: Any) -> int | None:
    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_sequence_pixel_spacing(ds: Dataset, sequence_name: str) -> list[float] | None:
    sequence = getattr(ds, sequence_name, None)
    if not sequence:
        return None

    for group in sequence:
        pixel_measures_sequence = getattr(group, "PixelMeasuresSequence", None)
        if not pixel_measures_sequence:
            continue

        for pixel_measures in pixel_measures_sequence:
            spacing = _extract_float_pair(getattr(pixel_measures, "PixelSpacing", None))
            if spacing is not None:
                logger.info(
                    "PixelSpacing not found directly, using %s.PixelMeasuresSequence: %s",
                    sequence_name,
                    spacing,
                )
                return spacing

    return None


def _extract_imager_pixel_spacing(ds: Dataset) -> list[float] | None:
    return _extract_float_pair(getattr(ds, "ImagerPixelSpacing", None))


def _extract_nominal_scanned_pixel_spacing(ds: Dataset) -> list[float] | None:
    return _extract_float_pair(getattr(ds, "NominalScannedPixelSpacing", None))


def _extract_pixel_spacing(ds: Dataset, file_path: str) -> tuple[list[float], str]:
    direct_spacing = _extract_float_pair(getattr(ds, "PixelSpacing", None))
    if direct_spacing is not None:
        return direct_spacing, "PixelSpacing"

    shared_spacing = _extract_sequence_pixel_spacing(ds, "SharedFunctionalGroupsSequence")
    if shared_spacing is not None:
        return shared_spacing, "SharedFunctionalGroupsSequence.PixelMeasuresSequence"

    per_frame_spacing = _extract_sequence_pixel_spacing(ds, "PerFrameFunctionalGroupsSequence")
    if per_frame_spacing is not None:
        return per_frame_spacing, "PerFrameFunctionalGroupsSequence.PixelMeasuresSequence"

    imager_spacing = _extract_imager_pixel_spacing(ds)
    if imager_spacing is not None:
        logger.info(
            "PixelSpacing not found directly, using ImagerPixelSpacing: %s",
            imager_spacing,
        )
        return imager_spacing, "ImagerPixelSpacing"

    nominal_scanned_spacing = _extract_nominal_scanned_pixel_spacing(ds)
    if nominal_scanned_spacing is not None:
        logger.info(
            "PixelSpacing not found directly, using NominalScannedPixelSpacing: %s",
            nominal_scanned_spacing,
        )
        return nominal_scanned_spacing, "NominalScannedPixelSpacing"

    logger.warning(
        "PixelSpacing not found in '%s'. Falling back to default [1.0, 1.0].",
        file_path,
    )
    return [1.0, 1.0], "default"


def _extract_study_date(ds: Dataset) -> tuple[str | None, str | None]:
    for field_name in DATE_FIELDS:
        value = _extract_text(ds, field_name)
        if value is not None:
            return value, field_name

    return None, None


def load_dicom(file_path: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a DICOM file and return the pixel array with extracted metadata."""
    ds = pydicom.dcmread(file_path)

    pixel_spacing, pixel_spacing_source = _extract_pixel_spacing(ds, file_path)
    imager_pixel_spacing = _extract_imager_pixel_spacing(ds)
    study_date, study_date_source = _extract_study_date(ds)

    pixel_array = ds.pixel_array.astype(np.float32)

    if hasattr(ds, "RescaleSlope") and hasattr(ds, "RescaleIntercept"):
        slope = float(ds.RescaleSlope)
        intercept = float(ds.RescaleIntercept)
        pixel_array = slope * pixel_array + intercept
        logger.debug("Applied rescale transform: slope=%.2f intercept=%.2f", slope, intercept)

    metadata: dict[str, Any] = {
        "pixel_spacing_mm": pixel_spacing,
        "pixel_spacing_source": pixel_spacing_source,
        "imager_pixel_spacing_mm": imager_pixel_spacing,
        "patient_id": _extract_text(ds, "PatientID"),
        "study_instance_uid": _extract_text(ds, "StudyInstanceUID"),
        "study_date": study_date,
        "study_date_source": study_date_source,
        "modality": _extract_text(ds, "Modality"),
        "sop_class_uid": _extract_text(ds, "SOPClassUID"),
        "photometric_interpretation": _extract_text(ds, "PhotometricInterpretation"),
        "samples_per_pixel": _extract_int(getattr(ds, "SamplesPerPixel", None)),
        "number_of_frames": _extract_int(getattr(ds, "NumberOfFrames", None)),
        "bits_allocated": getattr(ds, "BitsAllocated", None),
        "image_shape": list(pixel_array.shape),
    }

    logger.info(
        "DICOM loaded: %s | shape=%s | modality=%s | pixel_spacing=%s | source=%s",
        file_path,
        pixel_array.shape,
        metadata["modality"],
        pixel_spacing,
        pixel_spacing_source,
    )

    return pixel_array, metadata


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke-runner for the DICOM loader.")
    parser.add_argument("file_path", help="Path to a DICOM file")
    args = parser.parse_args(argv)

    try:
        _, metadata = load_dicom(args.file_path)
    except Exception:
        logger.exception("Failed to read DICOM '%s'", args.file_path)
        return 1

    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
