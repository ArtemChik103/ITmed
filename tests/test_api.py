from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from tests.helpers import build_test_dicom


def test_health_endpoint():
    from api.main import app

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "1.0.0"}


def test_list_plugins_endpoint():
    from api.main import app

    with TestClient(app) as client:
        response = client.get("/api/v1/plugins")

    assert response.status_code == 200
    payload = response.json()
    assert payload["plugins"][0]["name"] == "hip_dysplasia"


def test_analyze_returns_typed_metadata_and_plugin_fields(tmp_path: Path):
    from api.main import app

    dicom_path = build_test_dicom(
        tmp_path / "api_ok.dcm",
        imager_pixel_spacing=[0.199, 0.199],
        series_date="20240111",
    )

    with TestClient(app) as client:
        with dicom_path.open("rb") as file_obj:
            response = client.post(
                "/api/v1/analyze",
                params={"plugin_type": "hip_dysplasia", "mode": "education"},
                files={"file": ("api_ok.dcm", file_obj, "application/dicom")},
            )

    assert response.status_code == 200
    payload = response.json()
    assert payload["metadata"]["study_date"] == "20240111"
    assert payload["plugin_name"] == "hip_dysplasia"
    assert payload["plugin_version"] == "0.2.0"
    assert payload["validation_warnings"]
    assert payload["processing_time_ms"] >= 0


def test_analyze_rejects_non_dicom_payload(tmp_path: Path):
    from api.main import app

    bad_file = tmp_path / "broken.dcm"
    bad_file.write_text("this is not a dicom", encoding="utf-8")

    with TestClient(app) as client:
        with bad_file.open("rb") as file_obj:
            response = client.post(
                "/api/v1/analyze",
                files={"file": ("broken.dcm", file_obj, "application/dicom")},
            )

    assert response.status_code == 422
    assert "Не удалось прочитать DICOM файл" in response.json()["detail"]


def test_analyze_rejects_wrong_extension():
    from api.main import app

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/analyze",
            files={"file": ("wrong.txt", b"plain text", "text/plain")},
        )

    assert response.status_code == 422


def test_analyze_rejects_unknown_plugin(tmp_path: Path):
    from api.main import app

    dicom_path = build_test_dicom(tmp_path / "unknown_plugin.dcm", pixel_spacing=[0.5, 0.5])

    with TestClient(app) as client:
        with dicom_path.open("rb") as file_obj:
            response = client.post(
                "/api/v1/analyze",
                params={"plugin_type": "unknown_plugin"},
                files={"file": ("unknown_plugin.dcm", file_obj, "application/dicom")},
            )

    assert response.status_code == 404
