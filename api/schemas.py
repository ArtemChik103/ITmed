"""
api/schemas.py — Pydantic response models for the API.
"""
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Ответ эндпоинта /health."""

    status: str = Field(..., examples=["ok"])
    version: str = Field(..., examples=["1.0.0"])


class AnalysisResult(BaseModel):
    """
    Результат анализа медицинского снимка.
    Возвращается эндпоинтом POST /api/v1/analyze.
    """

    disease_detected: bool = Field(..., description="Обнаружена ли патология")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Уверенность модели [0.0 – 1.0]")
    metrics: dict[str, float] = Field(default_factory=dict, description="Дополнительные метрики")
    processing_time_ms: int = Field(..., ge=0, description="Время обработки в мс")
    message: str = Field(..., description="Сообщение от сервиса")
