"""Baseline hip dysplasia plugin used to validate the Phase 2 pipeline."""
from __future__ import annotations

from typing import Any

import numpy as np

from core.plugin_manager import AnalysisResult, PluginMetadata
from core.preprocessor import XRayPreprocessor, get_preprocessor
from plugins.base_plugin import BasePlugin


class HipDysplasiaPlugin(BasePlugin):
    """A deterministic pre-ML plugin that validates the end-to-end pipeline."""

    def __init__(self, preprocessor: XRayPreprocessor | None = None) -> None:
        self._metadata = PluginMetadata(
            name="hip_dysplasia",
            version="0.2.0",
            description=(
                "Baseline pelvis X-ray analysis plugin. "
                "Returns heuristic non-diagnostic output until the trained model is connected."
            ),
            supported_modalities=["DX", "CR", "XR", "RG", "RF"],
        )
        self._preprocessor = preprocessor or get_preprocessor()
        self._loaded = False

    def load_model(self) -> None:
        self._loaded = True

    def preprocess(self, image: np.ndarray, metadata: dict[str, Any]) -> np.ndarray:
        return self._preprocessor.preprocess(image, metadata)

    def analyze(
        self,
        image: np.ndarray,
        metadata: dict[str, Any],
        *,
        mode: str = "doctor",
    ) -> AnalysisResult:
        mean_intensity = float(np.mean(image))
        std_intensity = float(np.std(image))

        metrics = {
            "mean_intensity": round(mean_intensity, 6),
            "std_intensity": round(std_intensity, 6),
            "image_height": float(image.shape[0]),
            "image_width": float(image.shape[1]),
        }

        return AnalysisResult(
            disease_detected=False,
            confidence=0.5,
            metrics=metrics,
            keypoints=[],
            heatmap_url=None,
            metadata=metadata,
            message=(
                "Baseline heuristic plugin executed successfully. "
                f"Mode={mode}. Trained ML weights are not attached yet, "
                "so the result is non-diagnostic and предназначен для проверки pipeline."
            ),
            plugin_name=self._metadata.name,
            plugin_version=self._metadata.version,
        )

    def get_metadata(self) -> PluginMetadata:
        return self._metadata
