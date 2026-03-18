"""Pydantic response models for the FastAPI layer."""
from __future__ import annotations

from pydantic import BaseModel, Field

from core.plugin_manager import AnalysisResult, PluginMetadata


class HealthResponse(BaseModel):
    """Response returned by the health endpoint."""

    status: str = Field(..., examples=["ok"])
    version: str = Field(..., examples=["1.0.0"])


class PluginListResponse(BaseModel):
    """Wrapper for plugin metadata listing."""

    plugins: list[PluginMetadata] = Field(default_factory=list)
