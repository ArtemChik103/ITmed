from __future__ import annotations

from frontend.utils.clinical_report_builder import (
    build_clinical_report,
    build_pdf_clinical_report,
    geometry_available,
    geometry_metric_rows,
    geometry_reason,
)


def test_clinical_report_builder_handles_missing_geometry_honestly():
    result = {
        "disease_detected": False,
        "confidence": 0.31,
        "metrics": {
            "runtime_model_loaded": 1.0,
            "model_probability": 0.31,
            "model_threshold": 0.4,
            "keypoint_model_loaded": 1.0,
            "keypoint_count": 8.0,
            "geometry_available": 0.0,
            "geometry_confidence": 0.0,
            "geometry_metric_count": 0.0,
        },
        "keypoints": [(1.0, 2.0)] * 8,
    }

    report = build_clinical_report(result)
    pdf_report = build_pdf_clinical_report(result)

    assert geometry_available(result) is False
    assert "не были автоматически рассчитаны" in geometry_reason(result)
    assert "не рассчитано" in report
    assert "___" not in report
    assert "___" not in pdf_report


def test_clinical_report_builder_formats_available_geometry_metrics():
    result = {
        "disease_detected": True,
        "confidence": 0.83,
        "metrics": {
            "runtime_model_loaded": 1.0,
            "model_probability": 0.83,
            "model_threshold": 0.4,
            "keypoint_model_loaded": 1.0,
            "keypoint_count": 8.0,
            "geometry_available": 1.0,
            "geometry_confidence": 1.0,
            "geometry_metric_count": 6.0,
            "right_acetabular_angle_deg": 28.4,
            "left_acetabular_angle_deg": 22.1,
            "right_h_mm": 12.6,
            "left_h_mm": 11.4,
            "right_d_mm": 7.2,
            "left_d_mm": 6.8,
        },
        "keypoints": [(1.0, 2.0)] * 8,
    }

    rows = geometry_metric_rows(result, include_unavailable=False)
    report = build_clinical_report(result)

    assert geometry_available(result) is True
    assert len(rows) == 6
    assert ("Правый ацетабулярный угол", "28.4°") in rows
    assert "28.4°" in report
    assert "7.2 мм" in report
