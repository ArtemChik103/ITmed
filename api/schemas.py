"""Pydantic response models for the FastAPI layer."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from core.plugin_manager import PluginMetadata


class HealthResponse(BaseModel):
    """Response returned by the health endpoint."""

    status: str = Field(..., examples=["ok"])
    version: str = Field(..., examples=["1.0.0"])


class DicomMetadata(BaseModel):
    """Typed DICOM metadata returned by the API."""

    pixel_spacing_mm: list[float] = Field(
        default_factory=list,
        description="Pixel spacing in mm [row_spacing, col_spacing]",
    )
    pixel_spacing_source: (
        Literal[
            "PixelSpacing",
            "SharedFunctionalGroupsSequence.PixelMeasuresSequence",
            "PerFrameFunctionalGroupsSequence.PixelMeasuresSequence",
            "default",
        ]
        | None
    ) = Field(default=None)
    imager_pixel_spacing_mm: list[float] | None = Field(default=None)
    patient_id: str | None = Field(default=None)
    study_instance_uid: str | None = Field(default=None)
    study_date: str | None = Field(default=None)
    study_date_source: (
        Literal[
            "StudyDate",
            "SeriesDate",
            "AcquisitionDate",
            "ContentDate",
            "InstanceCreationDate",
        ]
        | None
    ) = Field(default=None)
    modality: str | None = Field(default=None)
    sop_class_uid: str | None = Field(default=None)
    photometric_interpretation: str | None = Field(default=None)
    samples_per_pixel: int | None = Field(default=None)
    number_of_frames: int | None = Field(default=None)
    bits_allocated: int | None = Field(default=None)
    image_shape: list[int] = Field(default_factory=list)


class AnalysisResult(BaseModel):
    """Typed response of POST /api/v1/analyze."""

    disease_detected: bool = Field(..., description="Whether pathology is detected")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    metrics: dict[str, float] = Field(default_factory=dict, description="Additional metrics")
    keypoints: list[tuple[float, float]] = Field(default_factory=list)
    heatmap_url: str | None = Field(default=None)
    processing_time_ms: int = Field(..., ge=0)
    message: str = Field(..., description="Service message")
    metadata: DicomMetadata = Field(default_factory=DicomMetadata)
    validation_warnings: list[str] = Field(default_factory=list)
    plugin_name: str = Field(..., description="Plugin identifier")
    plugin_version: str = Field(..., description="Plugin version")


class PluginListResponse(BaseModel):
    """Wrapper for plugin metadata listing."""

    plugins: list[PluginMetadata] = Field(default_factory=list)
