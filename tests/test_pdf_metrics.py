from __future__ import annotations

from frontend.utils.medical_text import get_detailed_report, get_pdf_report_text
from frontend.utils.pdf_export import generate_pdf_report


def test_medical_text_no_longer_contains_placeholder_gaps():
    result = {
        "disease_detected": True,
        "confidence": 0.79,
        "metrics": {
            "runtime_model_loaded": 1.0,
            "model_probability": 0.79,
            "model_threshold": 0.4,
            "keypoint_model_loaded": 1.0,
            "keypoint_count": 8.0,
            "geometry_available": 0.0,
            "geometry_confidence": 0.0,
            "geometry_metric_count": 0.0,
        },
        "metadata": {"modality": "DX"},
        "keypoints": [(1.0, 2.0)] * 8,
    }

    detailed = get_detailed_report(result)
    pdf_text = get_pdf_report_text(result)

    assert "___" not in detailed
    assert "___" not in pdf_text
    assert "Количественные геометрические показатели" in detailed


def test_generate_pdf_report_handles_missing_geometry_metrics():
    result = {
        "disease_detected": False,
        "confidence": 0.22,
        "metrics": {
            "runtime_model_loaded": 1.0,
            "model_probability": 0.22,
            "model_threshold": 0.4,
            "keypoint_model_loaded": 0.0,
            "keypoint_count": 0.0,
            "geometry_available": 0.0,
            "geometry_confidence": 0.0,
            "geometry_metric_count": 0.0,
        },
        "metadata": {
            "modality": "DX",
            "pixel_spacing_source": "PixelSpacing",
            "study_date": "20260101",
            "number_of_frames": 1,
        },
        "keypoints": [],
        "processing_time_ms": 12,
        "message": "ok",
        "validation_warnings": [],
        "plugin_name": "hip_dysplasia",
        "plugin_version": "0.2.0",
    }

    pdf_bytes = generate_pdf_report(result, "sample.dcm")

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
