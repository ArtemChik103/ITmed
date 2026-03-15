"""Mock responses used for Phase 1 smoke checks and fallback examples."""
from __future__ import annotations

from typing import Any

from core.plugin_manager import AnalysisResult


def build_mock_analysis_result(
    *,
    filename: str,
    metadata: dict[str, Any],
    processing_time_ms: int,
) -> AnalysisResult:
    """Return a typed mock payload compatible with the API response model."""
    return AnalysisResult(
        disease_detected=False,
        confidence=0.85,
        metrics={"sensitivity": 0.0, "specificity": 0.0, "auc": 0.0},
        processing_time_ms=processing_time_ms,
        message=(
            f"[MOCK] Файл '{filename}' обработан. "
            "Реальная ML-модель будет подключена в следующей фазе."
        ),
        metadata=metadata,
        plugin_name="mock",
        plugin_version="1.0.0",
    )
