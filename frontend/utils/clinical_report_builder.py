"""Safe formatting helpers for clinical/geometry sections in the UI and PDF."""
from __future__ import annotations

from html import escape
from typing import Any

from frontend.utils.report_formatting import (
    confidence_text,
    disease_label,
    keypoint_count,
    keypoint_model_loaded,
    model_probability,
    model_threshold,
    runtime_model_loaded,
)

GEOMETRY_METRIC_SPECS: tuple[tuple[str, str, str], ...] = (
    ("right_acetabular_angle_deg", "Правый ацетабулярный угол", "deg"),
    ("left_acetabular_angle_deg", "Левый ацетабулярный угол", "deg"),
    ("right_h_mm", "Правый показатель h", "mm"),
    ("left_h_mm", "Левый показатель h", "mm"),
    ("right_d_mm", "Правый показатель d", "mm"),
    ("left_d_mm", "Левый показатель d", "mm"),
)

GEOMETRY_UNAVAILABLE_REASON = (
    "Количественные геометрические показатели не были автоматически рассчитаны в текущей версии модели."
)
GEOMETRY_SEMANTICS_REASON = (
    "Семантика raw keypoints MTDDH пока недостаточно подтверждена для клинических вычислений."
)


def geometry_available(result: dict[str, Any]) -> bool:
    metrics = result.get("metrics") or {}
    return bool(int(round(float(metrics.get("geometry_available", 0.0)))))


def geometry_confidence(result: dict[str, Any]) -> float:
    metrics = result.get("metrics") or {}
    return float(metrics.get("geometry_confidence", 0.0))


def geometry_reason(result: dict[str, Any]) -> str:
    metrics = result.get("metrics") or {}
    if geometry_available(result):
        return "Геометрические показатели рассчитаны автоматически."
    if float(metrics.get("keypoint_model_loaded", 0.0)) <= 0.0:
        return "Анатомические ориентиры недоступны, поэтому геометрия не рассчитывалась."
    return f"{GEOMETRY_UNAVAILABLE_REASON} {GEOMETRY_SEMANTICS_REASON}"


def _format_metric_value(value: float | None, unit: str) -> str:
    if value is None:
        return "не рассчитано"
    suffix = "°" if unit == "deg" else " мм" if unit == "mm" else ""
    return f"{float(value):.1f}{suffix}"


def geometry_metric_rows(
    result: dict[str, Any],
    *,
    include_unavailable: bool = True,
) -> list[tuple[str, str]]:
    metrics = result.get("metrics") or {}
    rows: list[tuple[str, str]] = []
    for metric_key, label, unit in GEOMETRY_METRIC_SPECS:
        raw_value = metrics.get(metric_key)
        if raw_value is None and not include_unavailable:
            continue
        rows.append((label, _format_metric_value(float(raw_value), unit) if raw_value is not None else "не рассчитано"))
    return rows


def build_clinical_report(result: dict[str, Any]) -> str:
    """Build a safe plain-text report without placeholder clinical claims."""
    threshold = model_threshold(result)
    threshold_text = confidence_text(threshold) if threshold is not None else "не указан"
    anatomy_text = (
        f"Определено {keypoint_count(result)} анатомических ориентиров вспомогательной моделью."
        if runtime_model_loaded(result) and keypoint_model_loaded(result) and keypoint_count(result) > 0
        else "Анатомические ориентиры недоступны или не использовались в этом анализе."
    )
    lines = [
        "Общий обзор:",
        (
            f"ИИ-система сформировала classifier-first заключение: {disease_label(result)}. "
            f"Уверенность модели {confidence_text(model_probability(result))}, порог решения {threshold_text}."
        ),
        "",
        "Анатомические ориентиры:",
        anatomy_text,
        "Эти ориентиры используются только как explainability layer и не меняют итоговый диагноз.",
        "",
        "Количественная геометрия:",
        geometry_reason(result),
    ]

    metric_rows = geometry_metric_rows(result, include_unavailable=not geometry_available(result))
    for label, value in metric_rows:
        lines.append(f"- {label}: {value}")

    lines.extend(
        [
            "",
            "Заключение:",
            (
                "Автоматическое заключение требует обязательной верификации врачом-специалистом. "
                "Текущая версия не рассчитывает неподтвержденные клинические углы и расстояния из raw keypoints."
            ),
        ]
    )
    return "\n".join(lines)


def build_pdf_clinical_report(result: dict[str, Any]) -> str:
    """Build HTML-safe report text for ReportLab paragraphs."""
    paragraphs = [escape(paragraph) for paragraph in build_clinical_report(result).split("\n")]
    rendered: list[str] = []
    for paragraph in paragraphs:
        if not paragraph:
            rendered.append("<br/>")
            continue
        if paragraph.startswith("- "):
            rendered.append(f"• {paragraph[2:]}<br/>")
            continue
        rendered.append(f"{paragraph}<br/>")
    return "".join(rendered)
