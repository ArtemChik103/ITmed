"""FastAPI application entry point."""
from __future__ import annotations

import logging
import os
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import AnalysisResult, HealthResponse, PluginListResponse
from core.dicom_loader import load_dicom
from core.dicom_validator import DICOMValidator
from core.plugin_manager import PluginManager
from plugins.hip_dysplasia import HipDysplasiaPlugin

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".dcm", ".dicom"}
ALLOWED_MODES = {"doctor", "education"}
APP_VERSION = "1.0.0"


def build_plugin_manager() -> PluginManager:
    manager = PluginManager()
    manager.register(HipDysplasiaPlugin())
    return manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("ИТ+Мед 2026 API starting up...")
    app.state.plugin_manager = build_plugin_manager()
    app.state.dicom_validator = DICOMValidator()
    yield
    logger.info("API shutting down.")


app = FastAPI(
    title="ИТ+Мед 2026 API",
    description="AI System for Medical Imaging",
    version=APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health() -> HealthResponse:
    """Return service health status."""
    return HealthResponse(status="ok", version=APP_VERSION)


@app.get("/api/v1/plugins", response_model=PluginListResponse, tags=["Plugins"])
async def list_plugins() -> PluginListResponse:
    """Return registered plugin metadata."""
    manager: PluginManager = app.state.plugin_manager
    return PluginListResponse(plugins=manager.list_metadata())


@app.post("/api/v1/analyze", response_model=AnalysisResult, tags=["Analysis"])
async def analyze_image(
    file: UploadFile = File(...),
    plugin_type: str = Query("hip_dysplasia", pattern=r"^[a-z0-9_]+$"),
    mode: str = Query("doctor", pattern=r"^(doctor|education)$"),
) -> AnalysisResult:
    """Analyze a medical image using the selected plugin."""
    if mode not in ALLOWED_MODES:
        raise HTTPException(status_code=422, detail=f"Неподдерживаемый режим '{mode}'.")

    filename = file.filename or ""
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Неподдерживаемый формат файла '{ext}'. Ожидается .dcm или .dicom",
        )

    contents = await file.read()
    logger.info("Получен файл '%s', размер: %d байт", filename, len(contents))

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        image, metadata = load_dicom(tmp_path)
    except Exception as exc:
        logger.exception("Ошибка при чтении DICOM '%s'", filename)
        raise HTTPException(
            status_code=422,
            detail=f"Не удалось прочитать DICOM файл '{filename}': {exc}",
        ) from exc
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    validator: DICOMValidator = app.state.dicom_validator
    validation_report = validator.validate(image, metadata)
    if not validation_report.valid:
        first_error = validation_report.errors[0]
        raise HTTPException(
            status_code=422,
            detail=f"Невалидный DICOM файл '{filename}': {first_error.message}",
        )

    manager: PluginManager = app.state.plugin_manager
    try:
        result = manager.analyze(
            plugin_type,
            image,
            metadata,
            mode=mode,
            validation_warnings=validation_report.warning_messages(),
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return AnalysisResult.model_validate(result.model_dump())
