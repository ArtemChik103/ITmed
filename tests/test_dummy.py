"""Basic import smoke tests."""


def test_all_imports():
    from api.main import app  # noqa: F401
    from api.schemas import AnalysisResult, DicomMetadata, HealthResponse, PluginListResponse  # noqa: F401
    from core.dicom_loader import load_dicom  # noqa: F401
    from core.dicom_validator import DICOMValidator, ValidationIssue, ValidationReport  # noqa: F401
    from core.plugin_manager import AnalysisResult as CoreAnalysisResult  # noqa: F401
    from core.plugin_manager import IPlugin, PluginManager, PluginMetadata  # noqa: F401
    from core.plugin_registry import PluginRegistry  # noqa: F401
    from core.preprocessor import PreprocessingConfig, XRayPreprocessor  # noqa: F401
    from plugins.hip_dysplasia import HipDysplasiaPlugin  # noqa: F401
    from scripts.data_quality_check import scan_root  # noqa: F401
    from scripts.verify_id_format import collect_test_ids  # noqa: F401

    assert app is not None
