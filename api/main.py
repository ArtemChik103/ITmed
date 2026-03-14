"""
api/main.py — FastAPI application entry point.
Phase 1: health check + mock analyze endpoint.
"""
import logging
import time
import tempfile
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from core.dicom_loader import load_dicom

from api.schemas import AnalysisResult, HealthResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALLOWED_CONTENT_TYPES = {"application/octet-stream", "application/dicom", ""}
ALLOWED_EXTENSIONS = {".dcm", ".dicom"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🏥 ИТ+Мед 2026 API starting up...")
    yield
    logger.info("API shutting down.")


app = FastAPI(
    title="ИТ+Мед 2026 API",
    description="AI System for Medical Imaging — Phase 1",
    version="1.0.0",
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
    """Проверка работоспособности сервиса."""
    return HealthResponse(status="ok", version="1.0.0")


@app.post("/api/v1/analyze", response_model=AnalysisResult, tags=["Analysis"])
async def analyze(file: UploadFile = File(...)) -> AnalysisResult:
    """
    Анализ медицинского снимка (DICOM).

    Phase 1: возвращает mock-данные. В Phase 2 будет подключена ML-модель.
    """
    filename = file.filename or ""
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Неподдерживаемый формат файла '{ext}'. Ожидается .dcm или .dicom",
        )

    start = time.time()

    # Читаем файл (Phase 2 → передаём в DICOM Loader + ML Plugin)
    contents = await file.read()
    logger.info("Получен файл '%s', размер: %d байт", filename, len(contents))

    metadata = {}
    # Phase 1: Извлекаем метаданные для проверки работоспособности
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        _, metadata = load_dicom(tmp_path)
    except Exception as e:
        logger.error("Ошибка при чтении DICOM: %s", e)
    finally:
        os.remove(tmp_path)

    # Имитация обработки
    time.sleep(0.5)

    elapsed_ms = int((time.time() - start) * 1000)

    return AnalysisResult(
        disease_detected=False,
        confidence=0.85,
        metrics={"sensitivity": 0.0, "specificity": 0.0, "auc": 0.0},
        processing_time_ms=elapsed_ms,
        message=f"[MOCK] Файл '{filename}' обработан. ML-модель будет подключена в Phase 2.",
        metadata=metadata,
    )
