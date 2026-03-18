"""Baseline hip dysplasia plugin used to validate the Phase 2 pipeline."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from core.plugin_manager import AnalysisResult, PluginMetadata
from core.preprocessor import XRayPreprocessor, get_preprocessor
from plugins.base_plugin import BasePlugin
from plugins.hip_dysplasia.keypoint_runtime import (
    HipDysplasiaKeypointRuntime,
    resolve_keypoint_checkpoint_path,
)
from plugins.hip_dysplasia.model import HipDysplasiaEnsemble, resolve_model_manifest_path

logger = logging.getLogger(__name__)

GEOMETRY_UNAVAILABLE_MESSAGE = (
    "Quantitative geometry metrics were not auto-calculated because MTDDH raw keypoint semantics "
    "are not yet validated for clinical use."
)


class HipDysplasiaPlugin(BasePlugin):
    """A deterministic pre-ML plugin that validates the end-to-end pipeline."""

    def __init__(self, preprocessor: XRayPreprocessor | None = None) -> None:
        self._metadata = PluginMetadata(
            name="hip_dysplasia",
            version="0.2.0",
            description=(
                "Pelvis X-ray analysis plugin with Phase 3 classifier runtime and heuristic fallback."
            ),
            supported_modalities=["DX", "CR", "XR", "RG", "RF"],
        )
        self._preprocessor = preprocessor or get_preprocessor()
        self._loaded = False
        self._runtime_model: HipDysplasiaEnsemble | None = None
        self._keypoint_runtime: HipDysplasiaKeypointRuntime | None = None

    def load_model(self) -> None:
        manifest_path = resolve_model_manifest_path()
        if manifest_path is None:
            logger.info("Hip dysplasia model manifest not found. Using heuristic fallback.")
            self._runtime_model = None
        else:
            try:
                self._runtime_model = HipDysplasiaEnsemble(manifest_path)
                self._preprocessor = self._runtime_model.build_preprocessor()
                logger.info("Hip dysplasia ensemble loaded from %s", manifest_path)
            except Exception:
                logger.exception("Failed to load hip dysplasia ensemble. Falling back to heuristic mode.")
                self._runtime_model = None

        keypoint_checkpoint = resolve_keypoint_checkpoint_path()
        if keypoint_checkpoint is None:
            logger.info("Hip dysplasia keypoint checkpoint not configured. Explainability overlay disabled.")
            self._keypoint_runtime = None
        else:
            try:
                self._keypoint_runtime = HipDysplasiaKeypointRuntime(keypoint_checkpoint)
                logger.info("Hip dysplasia keypoint runtime loaded from %s", keypoint_checkpoint)
            except Exception:
                logger.exception("Failed to load hip dysplasia keypoint runtime. Explainability overlay disabled.")
                self._keypoint_runtime = None
        self._loaded = True

    def preprocess(self, image: np.ndarray, metadata: dict[str, Any]) -> np.ndarray:
        return self._preprocessor.preprocess(image, metadata)

    def _keypoint_metrics(self, *, keypoint_count: int) -> dict[str, float]:
        return {
            "keypoint_model_loaded": 1.0 if self._keypoint_runtime is not None else 0.0,
            "keypoint_count": float(keypoint_count),
        }

    def _geometry_metrics(self) -> dict[str, float]:
        return {
            "geometry_available": 0.0,
            "geometry_confidence": 0.0,
            "geometry_metric_count": 0.0,
        }

    def _scale_keypoints_to_original_image(
        self,
        keypoints: list[tuple[float, float]],
        *,
        processed_shape: tuple[int, int],
        metadata: dict[str, Any],
    ) -> list[tuple[float, float]]:
        image_shape = metadata.get("image_shape")
        if not isinstance(image_shape, list) or len(image_shape) < 2:
            return list(keypoints)

        original_height = float(image_shape[-2])
        original_width = float(image_shape[-1])
        if original_height <= 0 or original_width <= 0:
            return list(keypoints)

        processed_height = float(processed_shape[0])
        processed_width = float(processed_shape[1])
        scale_x = original_width / max(processed_width, 1.0)
        scale_y = original_height / max(processed_height, 1.0)

        scaled: list[tuple[float, float]] = []
        for x, y in keypoints:
            scaled_x = float(np.clip(x * scale_x, 0.0, max(original_width - 1.0, 0.0)))
            scaled_y = float(np.clip(y * scale_y, 0.0, max(original_height - 1.0, 0.0)))
            scaled.append((scaled_x, scaled_y))
        return scaled

    def _heuristic_result(
        self,
        image: np.ndarray,
        metadata: dict[str, Any],
        *,
        mode: str,
    ) -> AnalysisResult:
        mean_intensity = float(np.mean(image))
        std_intensity = float(np.std(image))

        metrics = {
            "mean_intensity": round(mean_intensity, 6),
            "std_intensity": round(std_intensity, 6),
            "image_height": float(image.shape[0]),
            "image_width": float(image.shape[1]),
            "runtime_model_loaded": 0.0,
        }
        metrics.update(self._keypoint_metrics(keypoint_count=0))
        metrics.update(self._geometry_metrics())

        return AnalysisResult(
            disease_detected=False,
            confidence=0.5,
            metrics=metrics,
            keypoints=[],
            heatmap_url=None,
            metadata=metadata,
            message=(
                "Heuristic fallback executed successfully. "
                f"Mode={mode}. Trained ML weights are unavailable, so the result is non-diagnostic."
            ),
            plugin_name=self._metadata.name,
            plugin_version=self._metadata.version,
        )

    def analyze(
        self,
        image: np.ndarray,
        metadata: dict[str, Any],
        *,
        mode: str = "doctor",
    ) -> AnalysisResult:
        if self._runtime_model is None:
            return self._heuristic_result(image, metadata, mode=mode)

        prediction = self._runtime_model.predict(image)
        keypoints: list[tuple[float, float]] = []
        if mode == "education" and self._keypoint_runtime is not None:
            keypoint_prediction = self._keypoint_runtime.predict(image)
            keypoints = self._scale_keypoints_to_original_image(
                keypoint_prediction.keypoints_xy,
                processed_shape=(int(image.shape[0]), int(image.shape[1])),
                metadata=metadata,
            )
        metrics = {
            "mean_intensity": round(float(np.mean(image)), 6),
            "std_intensity": round(float(np.std(image)), 6),
            "image_height": float(image.shape[0]),
            "image_width": float(image.shape[1]),
            "runtime_model_loaded": 1.0,
            "model_probability": float(prediction.probability),
            "model_threshold": float(prediction.threshold),
            "ensemble_folds": float(self._runtime_model.fold_count),
        }
        metrics.update(self._keypoint_metrics(keypoint_count=len(keypoints)))
        metrics.update(self._geometry_metrics())

        message = (
            "Phase 3 classifier ensemble executed successfully. "
            f"Mode={mode}. Decision threshold={prediction.threshold:.2f}."
        )
        if mode == "education" and keypoints:
            message = f"{message} {GEOMETRY_UNAVAILABLE_MESSAGE}"

        return AnalysisResult(
            disease_detected=prediction.disease_detected,
            confidence=float(prediction.probability),
            metrics=metrics,
            keypoints=keypoints,
            heatmap_url=None,
            metadata=metadata,
            message=message,
            plugin_name=self._metadata.name,
            plugin_version=self._metadata.version,
        )

    def get_metadata(self) -> PluginMetadata:
        return self._metadata
