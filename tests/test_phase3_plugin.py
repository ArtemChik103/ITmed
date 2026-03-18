from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from tests.helpers import build_test_dicom
from tests.phase3_helpers import (
    create_phase3_manifest,
    write_synthetic_checkpoint,
    write_synthetic_keypoint_checkpoint,
    write_synthetic_model_manifest,
)


def _build_runtime_manifest(tmp_path: Path) -> Path:
    experiment_dir = tmp_path / "experiment"
    checkpoint_path = write_synthetic_checkpoint(experiment_dir / "fold_0" / "best.pt")
    return write_synthetic_model_manifest(
        experiment_dir / "model_manifest.json",
        checkpoint_paths=[checkpoint_path],
        input_size=64,
        threshold=0.5,
    )


def test_plugin_loads_runtime_model_if_manifest_exists(tmp_path: Path, monkeypatch):
    from plugins.hip_dysplasia import HipDysplasiaPlugin
    from core.preprocessor import resolve_preprocessing_config

    manifest_path = _build_runtime_manifest(tmp_path)
    monkeypatch.setenv("HIP_DYSPLASIA_MODEL_MANIFEST", str(manifest_path))
    plugin = HipDysplasiaPlugin()
    plugin.load_model()

    result = plugin.analyze(np.ones((512, 512), dtype=np.float32), {"modality": "DX"})

    assert result.metrics["runtime_model_loaded"] == 1.0
    assert result.metrics["keypoint_model_loaded"] == 0.0
    assert result.metrics["geometry_available"] == 0.0
    assert result.plugin_name == "hip_dysplasia"
    assert plugin._preprocessor.config.target_size == (64, 64)
    assert plugin._preprocessor.config == resolve_preprocessing_config("default", target_size=(64, 64))


def test_plugin_falls_back_when_manifest_is_missing(tmp_path: Path, monkeypatch):
    from plugins.hip_dysplasia import HipDysplasiaPlugin

    missing_manifest = tmp_path / "missing_model_manifest.json"
    monkeypatch.setenv("HIP_DYSPLASIA_MODEL_MANIFEST", str(missing_manifest))
    plugin = HipDysplasiaPlugin()
    plugin.load_model()

    result = plugin.analyze(np.ones((512, 512), dtype=np.float32), {"modality": "DX"})

    assert result.metrics["runtime_model_loaded"] == 0.0
    assert result.metrics["keypoint_model_loaded"] == 0.0
    assert result.metrics["geometry_available"] == 0.0
    assert result.disease_detected is False


def test_plugin_returns_keypoints_only_in_education_mode(tmp_path: Path, monkeypatch):
    from plugins.hip_dysplasia import HipDysplasiaPlugin

    manifest_path = _build_runtime_manifest(tmp_path)
    keypoint_checkpoint = write_synthetic_keypoint_checkpoint(tmp_path / "keypoints" / "best.ckpt")
    monkeypatch.setenv("HIP_DYSPLASIA_MODEL_MANIFEST", str(manifest_path))
    monkeypatch.setenv("HIP_DYSPLASIA_KEYPOINT_CHECKPOINT", str(keypoint_checkpoint))
    plugin = HipDysplasiaPlugin()
    plugin.load_model()

    metadata = {"modality": "DX", "image_shape": [512, 512]}
    education_result = plugin.analyze(np.ones((512, 512), dtype=np.float32), metadata, mode="education")
    doctor_result = plugin.analyze(np.ones((512, 512), dtype=np.float32), metadata, mode="doctor")

    assert education_result.metrics["runtime_model_loaded"] == 1.0
    assert education_result.metrics["keypoint_model_loaded"] == 1.0
    assert education_result.metrics["keypoint_count"] == 8.0
    assert education_result.metrics["geometry_available"] == 0.0
    assert len(education_result.keypoints) == 8
    assert all(0.0 <= x <= 511.0 and 0.0 <= y <= 511.0 for x, y in education_result.keypoints)
    assert "Quantitative geometry metrics were not auto-calculated" in education_result.message

    assert doctor_result.metrics["keypoint_model_loaded"] == 1.0
    assert doctor_result.metrics["keypoint_count"] == 0.0
    assert doctor_result.metrics["geometry_available"] == 0.0
    assert doctor_result.keypoints == []


def test_plugin_ignores_broken_keypoint_checkpoint_and_keeps_classifier_response(tmp_path: Path, monkeypatch):
    from plugins.hip_dysplasia import HipDysplasiaPlugin

    manifest_path = _build_runtime_manifest(tmp_path)
    monkeypatch.setenv("HIP_DYSPLASIA_MODEL_MANIFEST", str(manifest_path))
    monkeypatch.setenv("HIP_DYSPLASIA_KEYPOINT_CHECKPOINT", str(tmp_path / "broken" / "missing.ckpt"))
    plugin = HipDysplasiaPlugin()
    plugin.load_model()

    result = plugin.analyze(np.ones((512, 512), dtype=np.float32), {"modality": "DX"}, mode="education")

    assert result.metrics["runtime_model_loaded"] == 1.0
    assert result.metrics["keypoint_model_loaded"] == 0.0
    assert result.metrics["keypoint_count"] == 0.0
    assert result.metrics["geometry_available"] == 0.0
    assert result.keypoints == []


def test_api_analyze_returns_valid_result_with_runtime_model(tmp_path: Path, monkeypatch):
    from api.main import app

    manifest_path = _build_runtime_manifest(tmp_path)
    monkeypatch.setenv("HIP_DYSPLASIA_MODEL_MANIFEST", str(manifest_path))
    dicom_path = build_test_dicom(tmp_path / "api_model.dcm", pixel_spacing=[0.2, 0.2], study_date="20260101")

    with TestClient(app) as client:
        with dicom_path.open("rb") as file_obj:
            response = client.post(
                "/api/v1/analyze",
                params={"plugin_type": "hip_dysplasia", "mode": "doctor"},
                files={"file": ("api_model.dcm", file_obj, "application/dicom")},
            )

    assert response.status_code == 200
    payload = response.json()
    assert payload["plugin_name"] == "hip_dysplasia"
    assert 0.0 <= payload["confidence"] <= 1.0
    assert payload["metrics"]["runtime_model_loaded"] == 1.0
    assert payload["metrics"]["keypoint_model_loaded"] == 0.0
    assert payload["metrics"]["geometry_available"] == 0.0


def test_api_analyze_keeps_valid_fallback_response_without_model(tmp_path: Path, monkeypatch):
    from api.main import app

    monkeypatch.setenv("HIP_DYSPLASIA_MODEL_MANIFEST", str(tmp_path / "missing_model_manifest.json"))
    dicom_path = build_test_dicom(tmp_path / "api_fallback.dcm", pixel_spacing=[0.2, 0.2], study_date="20260101")

    with TestClient(app) as client:
        with dicom_path.open("rb") as file_obj:
            response = client.post(
                "/api/v1/analyze",
                params={"plugin_type": "hip_dysplasia", "mode": "education"},
                files={"file": ("api_fallback.dcm", file_obj, "application/dicom")},
            )

    assert response.status_code == 200
    payload = response.json()
    assert payload["plugin_name"] == "hip_dysplasia"
    assert payload["metrics"]["runtime_model_loaded"] == 0.0
    assert payload["metrics"]["keypoint_model_loaded"] == 0.0
    assert payload["metrics"]["geometry_available"] == 0.0


def test_api_analyze_returns_keypoints_in_education_mode_when_runtime_is_available(tmp_path: Path, monkeypatch):
    from api.main import app

    manifest_path = _build_runtime_manifest(tmp_path)
    keypoint_checkpoint = write_synthetic_keypoint_checkpoint(tmp_path / "keypoints" / "best.ckpt")
    monkeypatch.setenv("HIP_DYSPLASIA_MODEL_MANIFEST", str(manifest_path))
    monkeypatch.setenv("HIP_DYSPLASIA_KEYPOINT_CHECKPOINT", str(keypoint_checkpoint))
    dicom_path = build_test_dicom(tmp_path / "api_keypoints.dcm", pixel_spacing=[0.2, 0.2], study_date="20260101")

    with TestClient(app) as client:
        with dicom_path.open("rb") as file_obj:
            response = client.post(
                "/api/v1/analyze",
                params={"plugin_type": "hip_dysplasia", "mode": "education"},
                files={"file": ("api_keypoints.dcm", file_obj, "application/dicom")},
            )

    assert response.status_code == 200
    payload = response.json()
    assert payload["metrics"]["runtime_model_loaded"] == 1.0
    assert payload["metrics"]["keypoint_model_loaded"] == 1.0
    assert payload["metrics"]["keypoint_count"] == 8.0
    assert payload["metrics"]["geometry_available"] == 0.0
    assert len(payload["keypoints"]) == 8
