"""Radiographic projection and clinical pose QA detector.

Distinguishes between standard anteroposterior (AP) pelvic radiographs and
functional/abduction views (Lauenstein / Frog-leg projection).
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

LAUENSTEIN_ADVISORY = (
    "Обнаружена укладка с функциональным отведением бёдер (положение по Лауэнштейну / Frog-leg view). "
    "Классические ориентиры Тённиса и Перкина валидированы для стандартной прямой проекции (AP). "
    "Физиологическое латеральное отведение бедренных костей в данной укладке является анатомической нормой позиционирования, а не вывихом."
)


def detect_xray_projection(
    image: np.ndarray,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Detect whether a pelvic radiograph is a standard AP view or a Lauenstein/Frog-leg view.

    Parameters
    ----------
    image : np.ndarray
        2D or 3D grayscale or RGB image array.
    metadata : dict[str, Any] | None
        DICOM metadata dictionary.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:
        - is_frog_leg: bool
        - projection_type: str ("standard_ap" | "lauenstein_frog_leg")
        - confidence: float (0.0 to 1.0)
        - clinical_advisory: str | None
    """
    metadata = metadata or {}

    # 1. Metadata check
    meta_text_fields = []
    for key in [
        "series_description",
        "protocol_name",
        "study_description",
        "image_comments",
        "body_part_examined",
        "view_position",
        "patient_position",
    ]:
        val = metadata.get(key)
        if val is not None:
            meta_text_fields.append(str(val).lower())

    combined_meta = " ".join(meta_text_fields)
    frog_keywords = ["lauenstein", "frog", "frog-leg", "отведение", "сидяч", "abduct"]
    for kw in frog_keywords:
        if kw in combined_meta:
            return {
                "is_frog_leg": True,
                "projection_type": "lauenstein_frog_leg",
                "confidence": 0.98,
                "clinical_advisory": LAUENSTEIN_ADVISORY,
                "detection_source": f"metadata:{kw}",
            }

    # 2. Image morphology check
    img_2d = image
    if img_2d.ndim == 3:
        if img_2d.shape[-1] in (1, 3, 4):
            img_2d = img_2d[..., 0]
        else:
            img_2d = img_2d[0]

    if img_2d.ndim != 2 or img_2d.size < 256:
        return {
            "is_frog_leg": False,
            "projection_type": "standard_ap",
            "confidence": 0.50,
            "clinical_advisory": None,
            "detection_source": "default",
        }

    h, w = img_2d.shape
    bot = img_2d[int(h * 0.70) :, :]
    mid_col = w // 2

    # Central perineal space: 38% to 62% of width in lower 30%
    c_start = int(w * 0.38)
    c_end = int(w * 0.62)
    center_band = bot[:, c_start:c_end]
    side_band_left = bot[:, : int(w * 0.25)]
    side_band_right = bot[:, int(w * 0.75) :]

    if center_band.size == 0 or side_band_left.size == 0 or side_band_right.size == 0:
        return {
            "is_frog_leg": False,
            "projection_type": "standard_ap",
            "confidence": 0.50,
            "clinical_advisory": None,
            "detection_source": "default",
        }

    c_mean = float(np.mean(center_band))
    s_mean = float((np.mean(side_band_left) + np.mean(side_band_right)) / 2.0)
    top_mean = float(np.mean(img_2d[: int(h * 0.35), :]))

    eps = 1e-4 if float(np.max(img_2d)) <= 5.0 else 1.0
    ratio_side_center = s_mean / max(eps, c_mean)
    ratio_top_center = top_mean / max(eps, c_mean)

    # In Lauenstein, the bottom center is dark perineal air / empty space while sides contain abducted thighs
    is_frog_leg = (ratio_side_center >= 1.01 and ratio_top_center >= 1.45)
    conf = min(0.95, max(0.55, 0.50 + (ratio_side_center - 1.0) * 2.0)) if is_frog_leg else 0.85

    return {
        "is_frog_leg": bool(is_frog_leg),
        "projection_type": "lauenstein_frog_leg" if is_frog_leg else "standard_ap",
        "confidence": round(conf, 2),
        "clinical_advisory": LAUENSTEIN_ADVISORY if is_frog_leg else None,
        "detection_source": "morphology",
    }
