"""
tests/test_dummy.py — Phase 1 baseline tests.
Verifies schemas, loader structure, and plugin manager.
"""
import pytest


class TestSchemas:
    def test_health_response(self):
        from api.schemas import HealthResponse
        resp = HealthResponse(status="ok", version="1.0.0")
        assert resp.status == "ok"
        assert resp.version == "1.0.0"

    def test_analysis_result(self):
        from api.schemas import AnalysisResult
        result = AnalysisResult(
            disease_detected=False,
            confidence=0.85,
            metrics={"auc": 0.9},
            processing_time_ms=500,
            message="test",
        )
        assert result.confidence == pytest.approx(0.85)
        assert result.disease_detected is False

    def test_analysis_result_confidence_bounds(self):
        from api.schemas import AnalysisResult
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            AnalysisResult(
                disease_detected=False,
                confidence=1.5,  # > 1.0 — должен упасть
                metrics={},
                processing_time_ms=0,
                message="bad",
            )


class TestPluginManager:
    def test_init(self):
        from core.plugin_manager import PluginManager
        pm = PluginManager()
        assert pm.list_plugins() == []

    def test_register_invalid(self):
        from core.plugin_manager import PluginManager
        pm = PluginManager()
        with pytest.raises(TypeError):
            pm.register("not_a_plugin")  # type: ignore

    def test_get_nonexistent(self):
        from core.plugin_manager import PluginManager
        pm = PluginManager()
        with pytest.raises(KeyError):
            pm.get_plugin("nonexistent")


class TestImports:
    def test_all_imports(self):
        """Smoke test: все модули импортируются без ошибок."""
        from api.schemas import AnalysisResult, HealthResponse  # noqa: F401
        from core.dicom_loader import load_dicom  # noqa: F401
        from core.plugin_manager import PluginInterface, PluginManager  # noqa: F401
        assert True
