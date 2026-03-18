"""Plugin contracts and registry used by the analysis pipeline."""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PluginMetadata(BaseModel):
    """Static metadata describing a plugin."""

    name: str = Field(..., description="Stable plugin identifier")
    version: str = Field(..., description="Plugin version")
    description: str = Field(..., description="Human-readable description")
    supported_modalities: list[str] = Field(
        default_factory=list,
        description="Supported DICOM modalities, for example DX or CR",
    )


class AnalysisResult(BaseModel):
    """Plugin analysis result shared between core, API and UI."""

    disease_detected: bool = Field(..., description="Whether pathology is detected")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence in [0, 1]")
    metrics: dict[str, float] = Field(default_factory=dict, description="Extra numeric metrics")
    keypoints: list[tuple[float, float]] = Field(
        default_factory=list,
        description="Detected keypoints in image coordinates",
    )
    heatmap_url: str | None = Field(default=None, description="Optional heatmap URL or artifact path")
    processing_time_ms: int = Field(default=0, ge=0, description="Processing time in milliseconds")
    metadata: dict[str, Any] = Field(default_factory=dict, description="DICOM metadata or plugin extras")
    message: str = Field(default="", description="Human-readable status message")
    validation_warnings: list[str] = Field(
        default_factory=list,
        description="Validation warnings that do not block analysis",
    )
    plugin_name: str = Field(default="", description="Plugin identifier")
    plugin_version: str = Field(default="", description="Plugin version used for the result")


class IPlugin(ABC):
    """Abstract plugin contract."""

    @abstractmethod
    def load_model(self) -> None:
        """Load plugin resources into memory."""

    @abstractmethod
    def preprocess(self, image: np.ndarray, metadata: dict[str, Any]) -> np.ndarray:
        """Prepare the image for analysis."""

    @abstractmethod
    def analyze(
        self,
        image: np.ndarray,
        metadata: dict[str, Any],
        *,
        mode: str = "doctor",
    ) -> AnalysisResult:
        """Run analysis on a preprocessed image."""

    @abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata."""


class PluginManager:
    """Registry and execution manager for analysis plugins."""

    def __init__(self) -> None:
        self._plugins: dict[str, IPlugin] = {}
        self._loaded_plugins: set[str] = set()
        logger.info("PluginManager initialized.")

    def register(self, plugin: IPlugin) -> None:
        metadata = plugin.get_metadata()
        if metadata.name in self._plugins:
            raise ValueError(f"Плагин '{metadata.name}' уже зарегистрирован.")

        self._plugins[metadata.name] = plugin
        logger.info("Plugin '%s' v%s registered.", metadata.name, metadata.version)

    def get(self, name: str) -> IPlugin:
        if name not in self._plugins:
            available = ", ".join(self.list_plugins()) or "нет"
            raise KeyError(f"Плагин '{name}' не найден. Доступные плагины: {available}.")
        return self._plugins[name]

    def list_plugins(self) -> list[str]:
        return sorted(self._plugins.keys())

    def list_metadata(self) -> list[PluginMetadata]:
        return [self._plugins[name].get_metadata() for name in self.list_plugins()]

    def ensure_loaded(self, name: str) -> IPlugin:
        plugin = self.get(name)
        if name not in self._loaded_plugins:
            plugin.load_model()
            self._loaded_plugins.add(name)
            logger.info("Plugin '%s' loaded.", name)
        return plugin

    def analyze(
        self,
        plugin_name: str,
        image: np.ndarray,
        metadata: dict[str, Any],
        *,
        mode: str = "doctor",
        validation_warnings: list[str] | None = None,
    ) -> AnalysisResult:
        plugin = self.ensure_loaded(plugin_name)

        started_at = time.perf_counter()
        processed_image = plugin.preprocess(image, metadata)
        result = plugin.analyze(processed_image, metadata, mode=mode)
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)

        warnings = list(validation_warnings or [])
        if result.validation_warnings:
            warnings.extend(result.validation_warnings)

        plugin_metadata = plugin.get_metadata()
        updates: dict[str, Any] = {
            "plugin_name": result.plugin_name or plugin_metadata.name,
            "plugin_version": result.plugin_version or plugin_metadata.version,
            "metadata": result.metadata or metadata,
            "validation_warnings": warnings,
        }

        if result.processing_time_ms == 0:
            updates["processing_time_ms"] = elapsed_ms

        return result.model_copy(update=updates)
