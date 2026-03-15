"""Validation helpers for DICOM files used by the X-ray pipeline."""
from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field


class ValidationIssue(BaseModel):
    """Single validation issue."""

    severity: Literal["error", "warning"] = Field(..., description="Issue severity")
    code: str = Field(..., description="Stable issue code")
    message: str = Field(..., description="Human-readable message")
    details: dict[str, Any] = Field(default_factory=dict, description="Extra machine-readable details")


class ValidationReport(BaseModel):
    """Aggregated validation result."""

    valid: bool = Field(..., description="True when there are no blocking errors")
    errors: list[ValidationIssue] = Field(default_factory=list)
    warnings: list[ValidationIssue] = Field(default_factory=list)

    def warning_messages(self) -> list[str]:
        return [issue.message for issue in self.warnings]


class DICOMValidator:
    """Validate X-ray DICOM metadata after loading."""

    def __init__(self, supported_modalities: set[str] | None = None) -> None:
        self.supported_modalities = supported_modalities or {"DX", "CR", "XR", "RG", "RF"}

    def validate(self, image: np.ndarray, metadata: dict[str, Any]) -> ValidationReport:
        errors: list[ValidationIssue] = []
        warnings: list[ValidationIssue] = []

        if image.size == 0:
            errors.append(
                ValidationIssue(
                    severity="error",
                    code="empty_image",
                    message="DICOM не содержит пиксельных данных.",
                )
            )

        if image.ndim not in (2, 3):
            errors.append(
                ValidationIssue(
                    severity="error",
                    code="invalid_shape",
                    message=f"Неподдерживаемая форма изображения: {list(image.shape)}.",
                    details={"image_shape": list(image.shape)},
                )
            )

        modality = (metadata.get("modality") or "").strip().upper()
        if not modality:
            warnings.append(
                ValidationIssue(
                    severity="warning",
                    code="missing_modality",
                    message="Тег Modality отсутствует. Предполагается X-ray изображение.",
                )
            )
        elif modality not in self.supported_modalities:
            errors.append(
                ValidationIssue(
                    severity="error",
                    code="unsupported_modality",
                    message=f"Модальность '{modality}' не поддерживается для X-ray pipeline.",
                    details={"modality": modality},
                )
            )

        samples_per_pixel = metadata.get("samples_per_pixel")
        if samples_per_pixel not in (None, 1):
            errors.append(
                ValidationIssue(
                    severity="error",
                    code="unsupported_samples_per_pixel",
                    message=(
                        "Поддерживаются только grayscale-изображения "
                        f"(SamplesPerPixel=1). Получено: {samples_per_pixel}."
                    ),
                    details={"samples_per_pixel": samples_per_pixel},
                )
            )

        number_of_frames = metadata.get("number_of_frames")
        if number_of_frames not in (None, 1):
            errors.append(
                ValidationIssue(
                    severity="error",
                    code="unsupported_multiframe",
                    message=(
                        "Текущий X-ray pipeline рассчитан на single-frame DICOM. "
                        f"Получено кадров: {number_of_frames}."
                    ),
                    details={"number_of_frames": number_of_frames},
                )
            )

        pixel_spacing = metadata.get("pixel_spacing_mm") or []
        if len(pixel_spacing) != 2:
            warnings.append(
                ValidationIssue(
                    severity="warning",
                    code="missing_pixel_spacing",
                    message="Размер пикселя не найден или задан некорректно.",
                )
            )

        pixel_spacing_source = metadata.get("pixel_spacing_source")
        if pixel_spacing_source == "default":
            warnings.append(
                ValidationIssue(
                    severity="warning",
                    code="default_pixel_spacing",
                    message="PixelSpacing не найден. Используется fallback [1.0, 1.0].",
                )
            )

        if not metadata.get("patient_id"):
            warnings.append(
                ValidationIssue(
                    severity="warning",
                    code="missing_patient_id",
                    message="PatientID отсутствует в DICOM.",
                )
            )

        if not metadata.get("study_instance_uid"):
            warnings.append(
                ValidationIssue(
                    severity="warning",
                    code="missing_study_instance_uid",
                    message="StudyInstanceUID отсутствует в DICOM.",
                )
            )

        if not metadata.get("study_date"):
            warnings.append(
                ValidationIssue(
                    severity="warning",
                    code="missing_study_date",
                    message="Дата исследования отсутствует в DICOM.",
                )
            )

        if not metadata.get("photometric_interpretation"):
            warnings.append(
                ValidationIssue(
                    severity="warning",
                    code="missing_photometric_interpretation",
                    message="PhotometricInterpretation отсутствует в DICOM.",
                )
            )

        return ValidationReport(valid=not errors, errors=errors, warnings=warnings)
