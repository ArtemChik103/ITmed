from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.helpers import build_test_dicom


class DummyPlugin:
    def __init__(self) -> None:
        self.load_calls = 0

    def load_model(self) -> None:
        self.load_calls += 1

    def preprocess(self, image, metadata):
        return image

    def analyze(self, image, metadata, *, mode: str = "doctor"):
        from core.plugin_manager import AnalysisResult

        return AnalysisResult(
            disease_detected=False,
            confidence=0.42,
            metrics={"pixels": float(image.size)},
            processing_time_ms=0,
            message=f"mode={mode}",
            metadata=metadata,
        )

    def get_metadata(self):
        from core.plugin_manager import PluginMetadata

        return PluginMetadata(
            name="dummy",
            version="1.0.0",
            description="dummy plugin",
            supported_modalities=["DX"],
        )


def test_load_dicom_uses_direct_pixel_spacing_and_metadata(tmp_path: Path):
    from core.dicom_loader import load_dicom

    dicom_path = build_test_dicom(
        tmp_path / "direct_spacing.dcm",
        pixel_spacing=[0.5, 0.7],
        patient_id="PAT-001",
        study_instance_uid="1.2.840.1",
        study_date="20260106",
    )

    image, metadata = load_dicom(str(dicom_path))

    assert image.dtype == np.float32
    assert metadata["pixel_spacing_mm"] == [0.5, 0.7]
    assert metadata["pixel_spacing_source"] == "PixelSpacing"
    assert metadata["patient_id"] == "PAT-001"
    assert metadata["study_instance_uid"] == "1.2.840.1"
    assert metadata["study_date_source"] == "StudyDate"
    assert metadata["photometric_interpretation"] == "MONOCHROME2"
    assert metadata["samples_per_pixel"] == 1
    assert metadata["number_of_frames"] is None


def test_load_dicom_uses_shared_functional_groups_spacing(tmp_path: Path):
    from core.dicom_loader import load_dicom

    dicom_path = build_test_dicom(tmp_path / "shared_spacing.dcm", shared_pixel_spacing=[0.33, 0.44])
    _, metadata = load_dicom(str(dicom_path))

    assert metadata["pixel_spacing_mm"] == [0.33, 0.44]
    assert metadata["pixel_spacing_source"] == "SharedFunctionalGroupsSequence.PixelMeasuresSequence"


def test_load_dicom_uses_per_frame_functional_groups_spacing(tmp_path: Path):
    from core.dicom_loader import load_dicom

    dicom_path = build_test_dicom(tmp_path / "per_frame_spacing.dcm", per_frame_pixel_spacing=[0.12, 0.13])
    _, metadata = load_dicom(str(dicom_path))

    assert metadata["pixel_spacing_mm"] == [0.12, 0.13]
    assert metadata["pixel_spacing_source"] == "PerFrameFunctionalGroupsSequence.PixelMeasuresSequence"


def test_load_dicom_defaults_spacing_and_logs_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    from core.dicom_loader import load_dicom

    dicom_path = build_test_dicom(tmp_path / "default_spacing.dcm", pixel_aspect_ratio=[4, 3])
    with caplog.at_level("WARNING"):
        _, metadata = load_dicom(str(dicom_path))

    assert metadata["pixel_spacing_mm"] == [1.0, 1.0]
    assert metadata["pixel_spacing_source"] == "default"
    assert "PixelSpacing not found" in caplog.text


def test_load_dicom_applies_rescale_transform(tmp_path: Path):
    from core.dicom_loader import load_dicom

    dicom_path = build_test_dicom(
        tmp_path / "rescale.dcm",
        pixel_array=np.array([[10, 20], [30, 40]], dtype=np.uint16),
        pixel_spacing=[1.0, 1.0],
        rescale_slope=2.0,
        rescale_intercept=-100.0,
    )

    image, _ = load_dicom(str(dicom_path))
    assert image.tolist() == [[-80.0, -60.0], [-40.0, -20.0]]


def test_dicom_validator_separates_errors_and_warnings():
    from core.dicom_validator import DICOMValidator

    validator = DICOMValidator()
    image = np.zeros((4, 4), dtype=np.float32)
    metadata = {
        "pixel_spacing_mm": [1.0, 1.0],
        "pixel_spacing_source": "default",
        "patient_id": None,
        "study_instance_uid": None,
        "study_date": None,
        "modality": "DX",
        "photometric_interpretation": "MONOCHROME2",
        "image_shape": [4, 4],
    }

    report = validator.validate(image, metadata)

    assert report.valid is True
    assert report.errors == []
    assert {warning.code for warning in report.warnings} >= {
        "default_pixel_spacing",
        "missing_patient_id",
        "missing_study_instance_uid",
        "missing_study_date",
    }


def test_dicom_validator_rejects_unsupported_modality():
    from core.dicom_validator import DICOMValidator

    validator = DICOMValidator()
    image = np.zeros((4, 4), dtype=np.float32)
    metadata = {
        "pixel_spacing_mm": [0.2, 0.2],
        "pixel_spacing_source": "PixelSpacing",
        "patient_id": "P-1",
        "study_instance_uid": "1.2.3",
        "study_date": "20260101",
        "modality": "CT",
        "photometric_interpretation": "MONOCHROME2",
        "image_shape": [4, 4],
    }

    report = validator.validate(image, metadata)

    assert report.valid is False
    assert report.errors[0].code == "unsupported_modality"


def test_dicom_validator_accepts_rf_single_frame_projection():
    from core.dicom_validator import DICOMValidator

    validator = DICOMValidator()
    image = np.zeros((4, 4), dtype=np.float32)
    metadata = {
        "pixel_spacing_mm": [0.111, 0.111],
        "pixel_spacing_source": "PixelSpacing",
        "patient_id": "P-1",
        "study_instance_uid": "1.2.3",
        "study_date": "20260101",
        "modality": "RF",
        "photometric_interpretation": "MONOCHROME2",
        "samples_per_pixel": 1,
        "number_of_frames": 1,
        "image_shape": [4, 4],
    }

    report = validator.validate(image, metadata)

    assert report.valid is True
    assert report.errors == []


def test_dicom_validator_rejects_multiframe_dicom():
    from core.dicom_validator import DICOMValidator

    validator = DICOMValidator()
    image = np.zeros((2, 4, 4), dtype=np.float32)
    metadata = {
        "pixel_spacing_mm": [0.111, 0.111],
        "pixel_spacing_source": "PixelSpacing",
        "patient_id": "P-1",
        "study_instance_uid": "1.2.3",
        "study_date": "20260101",
        "modality": "RF",
        "photometric_interpretation": "MONOCHROME2",
        "samples_per_pixel": 1,
        "number_of_frames": 2,
        "image_shape": [2, 4, 4],
    }

    report = validator.validate(image, metadata)

    assert report.valid is False
    assert {error.code for error in report.errors} >= {"unsupported_multiframe"}


def test_preprocessor_returns_normalized_resized_output():
    from core.preprocessor import XRayPreprocessor

    image = np.arange(100, dtype=np.float32).reshape(10, 10)
    metadata = {"photometric_interpretation": "MONOCHROME2"}

    preprocessor = XRayPreprocessor()
    processed = preprocessor.preprocess(image, metadata)

    assert processed.dtype == np.float32
    assert processed.shape == (512, 512)
    assert float(processed.min()) >= 0.0
    assert float(processed.max()) <= 1.0


def test_preprocessor_inverts_monochrome1():
    from core.preprocessor import XRayPreprocessor

    image = np.array([[0.0, 100.0], [200.0, 300.0]], dtype=np.float32)
    metadata = {"photometric_interpretation": "MONOCHROME1"}

    processed = XRayPreprocessor().preprocess(image, metadata)
    assert processed[0, 0] > processed[-1, -1]


def test_plugin_manager_registers_lists_and_lazy_loads():
    from core.plugin_manager import PluginManager

    plugin = DummyPlugin()
    manager = PluginManager()
    manager.register(plugin)

    result = manager.analyze("dummy", np.zeros((4, 4), dtype=np.float32), {"modality": "DX"})

    assert manager.list_plugins() == ["dummy"]
    assert plugin.load_calls == 1
    assert result.plugin_name == "dummy"
    assert result.processing_time_ms >= 0


def test_plugin_manager_rejects_duplicate_plugin():
    from core.plugin_manager import PluginManager

    manager = PluginManager()
    manager.register(DummyPlugin())
    with pytest.raises(ValueError):
        manager.register(DummyPlugin())
