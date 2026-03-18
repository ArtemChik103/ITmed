"""Formatting helpers for the Streamlit classifier-first UI."""
from __future__ import annotations

from typing import Any

from frontend.utils.keypoint_labels import raw_keypoint_order_text

MODE_LABELS = {"doctor": "Врач", "education": "Обучение"}
MODE_API_VALUES = {"Врач": "doctor", "Обучение": "education"}


def disease_label(result: dict[str, Any]) -> str:
    return "Патология" if result.get("disease_detected") else "Норма"


def disease_color(result: dict[str, Any]) -> str:
    if not runtime_model_loaded(result):
        return "#a16207"
    return "#b91c1c" if result.get("disease_detected") else "#166534"


def runtime_model_loaded(result: dict[str, Any]) -> bool:
    metrics = result.get("metrics") or {}
    return bool(int(round(float(metrics.get("runtime_model_loaded", 0.0)))))


def model_probability(result: dict[str, Any]) -> float:
    metrics = result.get("metrics") or {}
    return float(metrics.get("model_probability", result.get("confidence", 0.0)))


def model_threshold(result: dict[str, Any]) -> float | None:
    metrics = result.get("metrics") or {}
    value = metrics.get("model_threshold")
    if value is None:
        return None
    return float(value)


def ensemble_folds(result: dict[str, Any]) -> int | None:
    metrics = result.get("metrics") or {}
    value = metrics.get("ensemble_folds")
    if value is None:
        return None
    return int(round(float(value)))


def keypoint_model_loaded(result: dict[str, Any]) -> bool:
    metrics = result.get("metrics") or {}
    return bool(int(round(float(metrics.get("keypoint_model_loaded", 0.0)))))


def keypoint_count(result: dict[str, Any]) -> int:
    metrics = result.get("metrics") or {}
    return int(round(float(metrics.get("keypoint_count", len(result.get("keypoints") or [])))))


def confidence_text(value: float) -> str:
    return f"{value * 100:.1f}%"


def runtime_status_text(result: dict[str, Any]) -> str:
    if runtime_model_loaded(result):
        return "Используется обученная модель"
    return "Используется fallback-режим; результат недиагностический"


def keypoint_status_text(result: dict[str, Any]) -> str:
    if not runtime_model_loaded(result):
        return "Анатомический overlay отключен, потому что classifier runtime недоступен."
    if not keypoint_model_loaded(result):
        return "Анатомические ориентиры недоступны: optional keypoint checkpoint не загружен."
    if keypoint_count(result) <= 0:
        return "Анатомические ориентиры недоступны для этого анализа."
    return (
        f"Показаны {keypoint_count(result)} анатомических ориентиров из отдельной вспомогательной модели. "
        "Они не меняют итоговый диагноз."
    )


def metadata_summary(result: dict[str, Any]) -> list[tuple[str, str]]:
    metadata = result.get("metadata") or {}
    return [
        ("Modality", str(metadata.get("modality") or "не указана")),
        ("Pixel spacing source", str(metadata.get("pixel_spacing_source") or "нет данных")),
        ("Study date", str(metadata.get("study_date") or "нет данных")),
        ("Frames", str(metadata.get("number_of_frames") or 1)),
    ]


def doctor_summary(result: dict[str, Any]) -> str:
    label = disease_label(result)
    confidence = confidence_text(model_probability(result))
    threshold = model_threshold(result)
    threshold_text = confidence_text(threshold) if threshold is not None else "не указан"
    runtime_text = runtime_status_text(result)
    return (
        f"Заключение: {label}. Уверенность модели {confidence}, "
        f"порог решения {threshold_text}. {runtime_text}."
    )


def education_explanations(result: dict[str, Any]) -> list[str]:
    threshold = model_threshold(result)
    threshold_text = confidence_text(threshold) if threshold is not None else "не указан"
    explanations = [
        (
            f"Confidence {confidence_text(model_probability(result))} показывает, "
            "насколько модель склоняется к классу «патология»."
        ),
        (
            f"Threshold {threshold_text} задает границу, выше которой объект "
            "помечается как патологический."
        ),
        (
            "Fallback-режим нужен для отказоустойчивости интерфейса и API, "
            "но не является диагностическим заключением."
        ),
        (
            "Анатомические ориентиры, если они доступны, предсказываются отдельной вспомогательной моделью "
            "и используются только для визуализации anatomy layer."
        ),
        (
            "Binary verdict, confidence и threshold по-прежнему приходят только от classifier runtime. "
            "Keypoints в этой версии не меняют disease_detected."
        ),
        (
            "В текущей версии клинические углы и расстояния не рассчитываются автоматически, "
            "потому что raw keypoint semantics еще не подтверждены для clinical use."
        ),
        (
            "Heatmap, Hilgenreiner metrics и линии не добавляются. "
            "Raw internal keypoint order: " + raw_keypoint_order_text() + "."
        ),
    ]
    return explanations


def compact_metrics(result: dict[str, Any]) -> list[tuple[str, str]]:
    metrics = result.get("metrics") or {}
    items = [
        ("Диагноз", disease_label(result)),
        ("Уверенность", confidence_text(model_probability(result))),
        ("Порог", confidence_text(model_threshold(result)) if model_threshold(result) is not None else "n/a"),
        ("Runtime", "model" if runtime_model_loaded(result) else "fallback"),
    ]
    folds = ensemble_folds(result)
    if folds is not None:
        items.append(("Фолды", str(folds)))
    if result.get("processing_time_ms") is not None:
        items.append(("Время", f"{int(result['processing_time_ms'])} мс"))
    return items


def history_entry(filename: str, mode: str, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "filename": filename,
        "mode": MODE_LABELS.get(mode, mode),
        "disease_detected": bool(result.get("disease_detected")),
        "confidence": round(model_probability(result), 4),
        "runtime_model_loaded": int(runtime_model_loaded(result)),
        "short_summary": doctor_summary(result),
        "json_summary": {
            "message": result.get("message"),
            "metrics": result.get("metrics") or {},
            "metadata": result.get("metadata") or {},
            "validation_warnings": result.get("validation_warnings") or [],
        },
    }
