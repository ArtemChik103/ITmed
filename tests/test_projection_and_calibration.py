"""Unit and integration tests for radiographic projection detection, PDF Cyrillic rendering, and clinical grading."""
from __future__ import annotations

import numpy as np
import pytest

from core.projection_detector import LAUENSTEIN_ADVISORY, detect_xray_projection
from frontend.utils.pdf_export import _register_fonts, generate_pdf_report
from plugins.hip_dysplasia.plugin import HipDysplasiaPlugin


def test_projection_detector_metadata():
    """Verify projection detector identifies frog-leg view from metadata."""
    res_ap = detect_xray_projection(np.ones((100, 100)), {"series_description": "Pelvis AP"})
    assert res_ap["is_frog_leg"] is False
    assert res_ap["projection_type"] == "standard_ap"

    res_fl = detect_xray_projection(np.ones((100, 100)), {"protocol_name": "Lauenstein frog-leg"})
    assert res_fl["is_frog_leg"] is True
    assert res_fl["projection_type"] == "lauenstein_frog_leg"
    assert "Лауэнштейн" in res_fl["clinical_advisory"]


def test_projection_detector_morphology():
    """Verify morphology check distinguishes AP vs frog-leg leg spread."""
    # Synthetic frog-leg image: dark perineum in bottom center, dense sides, bright top
    img_fl = np.ones((500, 500), dtype=np.float32) * 1500
    img_fl[:175, :] = 2000 # bright top
    img_fl[350:, 190:310] = 800 # dark perineal gap
    img_fl[350:, :125] = 1300 # side thighs
    img_fl[350:, 375:] = 1300 # side thighs

    res_fl = detect_xray_projection(img_fl, {})
    assert res_fl["is_frog_leg"] is True
    assert res_fl["projection_type"] == "lauenstein_frog_leg"


def test_pdf_cyrillic_font_registered():
    """Verify font registration returns unicode font supporting Cyrillic."""
    regular, bold = _register_fonts()
    assert regular in ("DejaVuSans", "Arial", "SysDejaVu", "SysLiberation")
    assert bold in ("DejaVuSans-Bold", "Arial-Bold", "SysDejaVu-Bold", "SysLiberation-Bold")


def test_pdf_generation_with_frog_leg():
    """Verify PDF generates successfully with Cyrillic text and frog-leg advisory."""
    mock_result = {
        "disease_detected": False,
        "confidence": 0.66,
        "metrics": {
            "model_probability": 0.66,
            "tonnis_grade": 0.0,
            "reimers_index_pct": 16.5,
            "acetabular_angle_deg": 21.5,
            "peak_contact_stress_mpa": 1.35,
            "is_frog_leg": 1.0,
            "ensemble_folds": 3.0,
        },
        "metadata": {
            "is_frog_leg": True,
            "projection_type": "lauenstein_frog_leg",
        }
    }
    pdf_bytes = generate_pdf_report(mock_result, "test_frog.dcm")
    assert len(pdf_bytes) > 5000
    assert b"%PDF" in pdf_bytes[:10]


def test_safe_tonnis_grading():
    """Verify moderate probabilities do not falsely jump to Tönnis Grade 3 (dislocation)."""
    plugin = HipDysplasiaPlugin()
    # Dummy AP image
    dummy = np.ones((400, 400), dtype=np.float32) * 1000
    res = plugin.analyze(dummy, {}, mode="doctor")
    # Heuristic/default should not trigger dislocation on neutral image
    assert res.metrics["tonnis_grade"] < 3.0


def test_projection_detector_normalized_scale():
    """Verify projection detector morphology works on normalized [0, 1] images."""
    img_fl = np.ones((500, 500), dtype=np.float32) * 0.75
    img_fl[:175, :] = 1.0 # bright top
    img_fl[350:, 190:310] = 0.40 # dark perineal gap
    img_fl[350:, :125] = 0.65 # side thighs
    img_fl[350:, 375:] = 0.65 # side thighs

    res_fl = detect_xray_projection(img_fl, {})
    assert res_fl["is_frog_leg"] is True
    assert res_fl["projection_type"] == "lauenstein_frog_leg"


def test_keypoint_overlay_font_loading():
    """Verify keypoint overlay font loader loads a scalable TTF font with Cyrillic support."""
    from frontend.components.keypoint_overlay import _load_font

    font = _load_font(20)
    assert font is not None
    # Ensure it is a FreeTypeFont or TrueType font with bbox method
    assert hasattr(font, "getbbox")
    bbox = font.getbbox("Тест Тённис")
    assert bbox[2] > bbox[0]

