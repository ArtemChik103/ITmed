"""Shared helpers for Phase 5 submission and demo tooling."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from core.dicom_loader import load_dicom
from plugins.hip_dysplasia.model import HipDysplasiaEnsemble, resolve_model_manifest_path
from train.aggregation import aggregate_probability

REPO_ROOT = Path(__file__).resolve().parents[1]
DICOM_EXTENSIONS = {".dcm", ".dicom"}
DEFAULT_PLUGIN_TYPE = "hip_dysplasia"
DEFAULT_ANALYSIS_MODE = "doctor"
DEFAULT_AGGREGATION_METHOD = "max"


@dataclass(slots=True)
class TestObject:
    """Single submission object resolved from the mixed `test_done` layout."""

    object_id: str
    source_path: Path
    dicom_paths: list[Path]


@dataclass(slots=True)
class ImagePrediction:
    """Inference result for one DICOM image."""

    path: Path
    probability: float
    threshold: float
    disease_detected: bool
    runtime_model_loaded: bool
    metadata: dict[str, Any]
    message: str
    processing_time_ms: int = 0
    fold_probabilities: list[float] | None = None
    validation_warnings: list[str] | None = None
    raw_response: dict[str, Any] | None = None


@dataclass(slots=True)
class ObjectPrediction:
    """Object-level inference result used for CSV, screenshots and demo summaries."""

    object_id: str
    probability: float
    threshold: float
    disease_detected: bool
    runtime_model_loaded: bool
    num_images_in_object: int
    aggregation_method: str
    fold_probabilities: list[float] | None
    message: str
    metadata: dict[str, Any]
    validation_warnings: list[str]
    image_predictions: list[ImagePrediction]

    def csv_class(self) -> int:
        return int(self.disease_detected)

    def detailed_row(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "probability": round(float(self.probability), 6),
            "threshold": round(float(self.threshold), 6),
            "fold_probs": (
                "|".join(f"{value:.6f}" for value in self.fold_probabilities)
                if self.fold_probabilities
                else ""
            ),
            "num_images_in_object": int(self.num_images_in_object),
            "final_class": int(self.disease_detected),
            "runtime_model_loaded": int(self.runtime_model_loaded),
            "aggregation_method": self.aggregation_method,
            "message": self.message,
        }


def load_default_aggregation_method() -> str:
    """Resolve the canonical object-level aggregation method without analysis-side artifacts."""
    method = os.getenv("HIP_DYSPLASIA_OBJECT_AGGREGATION", DEFAULT_AGGREGATION_METHOD).strip().lower()
    return method or DEFAULT_AGGREGATION_METHOD


def is_dicom_path(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in DICOM_EXTENSIONS


def collect_test_objects(test_root: Path) -> list[TestObject]:
    """Resolve submission objects from a mixed root of DICOM files and folders."""
    if not test_root.exists():
        raise FileNotFoundError(f"Path does not exist: {test_root}")

    objects: list[TestObject] = []
    for item in sorted(test_root.iterdir(), key=lambda value: value.name.lower()):
        if item.is_dir():
            dicom_paths = sorted(
                path for path in item.rglob("*") if is_dicom_path(path)
            )
            if not dicom_paths:
                continue
            objects.append(TestObject(object_id=item.name, source_path=item, dicom_paths=dicom_paths))
            continue

        if is_dicom_path(item):
            objects.append(TestObject(object_id=item.stem, source_path=item, dicom_paths=[item]))

    return objects


def ensure_single_frame(image: np.ndarray) -> np.ndarray:
    """Collapse a DICOM pixel array into a single 2D frame."""
    if image.ndim == 2:
        return image.astype(np.float32, copy=False)
    if image.ndim == 3:
        if image.shape[0] == 1:
            return image[0].astype(np.float32, copy=False)
        if image.shape[-1] == 1:
            return image[..., 0].astype(np.float32, copy=False)
        if image.shape[-1] in {3, 4}:
            return np.mean(image[..., :3], axis=2, dtype=np.float32)
        frame_index = int(image.shape[0] // 2)
        return image[frame_index].astype(np.float32, copy=False)
    if image.ndim == 4:
        return ensure_single_frame(image[image.shape[0] // 2])
    raise ValueError(f"Unsupported image shape: {list(image.shape)}")


def load_dicom_for_inference(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a DICOM and normalize it to a single-frame grayscale array."""
    image, metadata = load_dicom(str(path))
    return ensure_single_frame(image), metadata


def normalize_display_image(image: np.ndarray, metadata: dict[str, Any]) -> np.ndarray:
    """Convert a grayscale image into displayable uint8 pixels."""
    frame = ensure_single_frame(image).astype(np.float32, copy=False)
    lower = float(np.percentile(frame, 1.0))
    upper = float(np.percentile(frame, 99.0))
    if upper <= lower:
        lower = float(frame.min())
        upper = float(frame.max())

    if upper <= lower:
        normalized = np.zeros_like(frame, dtype=np.float32)
    else:
        normalized = np.clip(frame, lower, upper)
        normalized = (normalized - lower) / (upper - lower)

    if metadata.get("photometric_interpretation") == "MONOCHROME1":
        normalized = 1.0 - normalized

    return np.clip(normalized * 255.0, 0.0, 255.0).astype(np.uint8)


def load_dicom_preview(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a DICOM preview image suitable for UI or screenshots."""
    image, metadata = load_dicom(str(path))
    return normalize_display_image(image, metadata), metadata


def _select_evenly_spaced_indices(total: int, limit: int) -> list[int]:
    if total <= limit:
        return list(range(total))

    if limit <= 1:
        return [0]

    return sorted(
        {
            min(total - 1, int(round(position)))
            for position in np.linspace(0, total - 1, num=limit)
        }
    )


def select_representative_dicoms(dicom_paths: list[Path], *, limit: int = 4) -> list[Path]:
    """Pick up to four representative paths for object screenshots."""
    indices = _select_evenly_spaced_indices(len(dicom_paths), limit)
    return [dicom_paths[index] for index in indices]


def _label_text(prediction: ObjectPrediction) -> str:
    return "ПАТОЛОГИЯ" if prediction.disease_detected else "НОРМА"


def _banner_color(prediction: ObjectPrediction) -> tuple[int, int, int]:
    if not prediction.runtime_model_loaded:
        return (184, 134, 11)
    if prediction.disease_detected:
        return (183, 28, 28)
    return (46, 125, 50)


def _text_color(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    brightness = (rgb[0] * 299 + rgb[1] * 587 + rgb[2] * 114) / 1000
    return (15, 23, 42) if brightness > 150 else (255, 255, 255)


def _load_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    for font_name in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    box: tuple[int, int, int, int],
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    fill: tuple[int, int, int],
    line_spacing: int = 8,
) -> None:
    left, top, right, bottom = box
    max_width = max(1, right - left)
    words = text.split()
    lines: list[str] = []
    current = ""

    for word in words:
        candidate = f"{current} {word}".strip()
        if draw.textlength(candidate, font=font) <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)

    y = top
    for line in lines:
        bbox = draw.textbbox((left, y), line, font=font)
        if bbox[3] > bottom:
            break
        draw.text((left, y), line, font=font, fill=fill)
        y = bbox[3] + line_spacing


def _build_contact_sheet(
    preview_images: list[np.ndarray],
    *,
    width: int,
    height: int,
    background: tuple[int, int, int],
) -> Image.Image:
    canvas = Image.new("RGB", (width, height), background)
    if not preview_images:
        return canvas

    count = len(preview_images)
    if count == 1:
        grid = (1, 1)
    elif count == 2:
        grid = (2, 1)
    else:
        grid = (2, 2)

    columns, rows = grid
    gutter = 18
    cell_width = (width - gutter * (columns + 1)) // columns
    cell_height = (height - gutter * (rows + 1)) // rows

    for index, image_array in enumerate(preview_images[: columns * rows]):
        row = index // columns
        column = index % columns
        x = gutter + column * (cell_width + gutter)
        y = gutter + row * (cell_height + gutter)

        grayscale = Image.fromarray(image_array, mode="L").convert("RGB")
        grayscale.thumbnail((cell_width, cell_height))
        frame = Image.new("RGB", (cell_width, cell_height), (17, 24, 39))
        offset_x = (cell_width - grayscale.width) // 2
        offset_y = (cell_height - grayscale.height) // 2
        frame.paste(grayscale, (offset_x, offset_y))

        canvas.paste(frame, (x, y))

    return canvas


def render_prediction_screenshot(
    prediction: ObjectPrediction,
    preview_images: list[np.ndarray],
    *,
    canvas_size: tuple[int, int] = (1440, 900),
) -> Image.Image:
    """Render a single screenshot with contact sheet and textual summary."""
    width, height = canvas_size
    background = (246, 241, 232)
    ink = (17, 24, 39)
    muted = (77, 86, 99)
    accent = _banner_color(prediction)
    accent_text = _text_color(accent)

    canvas = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(36)
    subtitle_font = _load_font(20)
    body_font = _load_font(24)
    small_font = _load_font(18)

    draw.rounded_rectangle((28, 24, width - 28, height - 24), radius=28, fill=(255, 251, 244))
    draw.rounded_rectangle((48, 46, width - 48, 146), radius=24, fill=accent)
    draw.text((72, 72), _label_text(prediction), font=title_font, fill=accent_text)
    draw.text(
        (72, 110),
        f"Объект {prediction.object_id} · classifier-first runtime",
        font=subtitle_font,
        fill=accent_text,
    )

    sheet = _build_contact_sheet(
        preview_images,
        width=760,
        height=height - 220,
        background=(232, 226, 216),
    )
    canvas.paste(sheet, (52, 176))

    right_panel = (844, 176, width - 56, height - 56)
    draw.rounded_rectangle(right_panel, radius=28, fill=(248, 244, 236), outline=(224, 218, 208))

    label = "Используется обученная модель" if prediction.runtime_model_loaded else "Fallback-режим"
    footer = (
        "classifier-first runtime"
        if prediction.runtime_model_loaded
        else "fallback mode · результат недиагностический"
    )
    confidence_text = f"{prediction.probability * 100:.1f}%"
    threshold_text = f"{prediction.threshold * 100:.1f}%"
    modality = prediction.metadata.get("modality") or "не указана"
    spacing_source = prediction.metadata.get("pixel_spacing_source") or "нет данных"
    warnings_text = (
        "; ".join(prediction.validation_warnings[:3])
        if prediction.validation_warnings
        else "Нет критичных предупреждений."
    )

    left, top, right, bottom = right_panel
    draw.text((left + 30, top + 30), label, font=subtitle_font, fill=accent)
    draw.text((left + 30, top + 76), f"Уверенность: {confidence_text}", font=body_font, fill=ink)
    draw.text((left + 30, top + 122), f"Порог решения: {threshold_text}", font=body_font, fill=ink)
    draw.text(
        (left + 30, top + 168),
        f"Снимков в объекте: {prediction.num_images_in_object}",
        font=body_font,
        fill=ink,
    )
    draw.text((left + 30, top + 214), f"Modality: {modality}", font=body_font, fill=ink)
    draw.text(
        (left + 30, top + 260),
        f"Источник pixel spacing: {spacing_source}",
        font=small_font,
        fill=muted,
    )

    draw.rounded_rectangle((left + 30, top + 320, right - 30, top + 390), radius=18, fill=(255, 250, 242))
    _draw_wrapped_text(
        draw,
        prediction.message,
        box=(left + 48, top + 340, right - 48, top + 384),
        font=small_font,
        fill=muted,
    )

    draw.text((left + 30, top + 430), "Предупреждения", font=subtitle_font, fill=ink)
    _draw_wrapped_text(
        draw,
        warnings_text,
        box=(left + 30, top + 466, right - 30, top + 580),
        font=small_font,
        fill=muted,
    )

    if prediction.fold_probabilities:
        fold_text = ", ".join(f"{value:.2f}" for value in prediction.fold_probabilities)
    else:
        fold_text = "недоступно"
    draw.text((left + 30, top + 622), f"Fold probabilities: {fold_text}", font=small_font, fill=muted)
    draw.text((left + 30, bottom - 56), footer, font=subtitle_font, fill=accent)
    return canvas


class LocalRuntimeAnalyzer:
    """Direct local runtime inference without the HTTP API."""

    def __init__(
        self,
        *,
        plugin_type: str = DEFAULT_PLUGIN_TYPE,
        manifest_path: str | Path | None = None,
        aggregation_method: str | None = None,
    ) -> None:
        if plugin_type != DEFAULT_PLUGIN_TYPE:
            raise ValueError(f"Unsupported local runtime plugin '{plugin_type}'.")

        self.plugin_type = plugin_type
        self.aggregation_method = aggregation_method or load_default_aggregation_method()
        self.resolved_manifest_path = resolve_model_manifest_path(manifest_path)
        self.ensemble: HipDysplasiaEnsemble | None = None
        self.preprocessor = None

        if self.resolved_manifest_path is not None:
            self.ensemble = HipDysplasiaEnsemble(self.resolved_manifest_path)
            self.preprocessor = self.ensemble.build_preprocessor()

    @property
    def runtime_model_loaded(self) -> bool:
        return self.ensemble is not None

    def predict_image(self, path: Path) -> ImagePrediction:
        image, metadata = load_dicom_for_inference(path)

        if self.ensemble is None or self.preprocessor is None:
            return ImagePrediction(
                path=path,
                probability=0.5,
                threshold=0.5,
                disease_detected=False,
                runtime_model_loaded=False,
                metadata=metadata,
                message=(
                    "Heuristic fallback executed successfully. "
                    "Trained ML weights are unavailable, so the result is non-diagnostic."
                ),
                fold_probabilities=None,
                validation_warnings=[],
            )

        processed = self.preprocessor.preprocess(image, metadata)
        prediction = self.ensemble.predict(processed)
        return ImagePrediction(
            path=path,
            probability=float(prediction.probability),
            threshold=float(prediction.threshold),
            disease_detected=bool(prediction.disease_detected),
            runtime_model_loaded=True,
            metadata=metadata,
            message=(
                "Phase 3 classifier ensemble executed successfully. "
                f"Decision threshold={prediction.threshold:.2f}."
            ),
            fold_probabilities=[float(value) for value in prediction.fold_probabilities],
            validation_warnings=[],
        )

    def predict_object(self, test_object: TestObject) -> ObjectPrediction:
        image_predictions = [self.predict_image(path) for path in test_object.dicom_paths]
        probabilities = [prediction.probability for prediction in image_predictions]
        probability = aggregate_probability(probabilities, method=self.aggregation_method)
        runtime_loaded = all(prediction.runtime_model_loaded for prediction in image_predictions)
        threshold = float(image_predictions[0].threshold if image_predictions else 0.5)

        if runtime_loaded:
            disease_detected = probability >= threshold
            fold_probabilities = None
            if image_predictions and image_predictions[0].fold_probabilities:
                fold_count = len(image_predictions[0].fold_probabilities or [])
                aggregated_folds = []
                for fold_index in range(fold_count):
                    fold_values = [
                        prediction.fold_probabilities[fold_index]
                        for prediction in image_predictions
                        if prediction.fold_probabilities is not None
                    ]
                    aggregated_folds.append(
                        aggregate_probability(fold_values, method=self.aggregation_method)
                    )
                fold_probabilities = aggregated_folds
            message = (
                "Classifier-first runtime aggregated object-level prediction "
                f"using '{self.aggregation_method}'."
            )
        else:
            disease_detected = False
            fold_probabilities = None
            message = (
                "Fallback mode used for object-level aggregation. "
                "Result is non-diagnostic and should only keep the pipeline operational."
            )

        warnings = [
            warning
            for prediction in image_predictions
            for warning in (prediction.validation_warnings or [])
        ]
        metadata = dict(image_predictions[0].metadata) if image_predictions else {}

        return ObjectPrediction(
            object_id=test_object.object_id,
            probability=float(probability),
            threshold=threshold,
            disease_detected=disease_detected,
            runtime_model_loaded=runtime_loaded,
            num_images_in_object=len(test_object.dicom_paths),
            aggregation_method=self.aggregation_method,
            fold_probabilities=fold_probabilities,
            message=message,
            metadata=metadata,
            validation_warnings=warnings,
            image_predictions=image_predictions,
        )


class ApiRuntimeAnalyzer:
    """HTTP API-backed analyzer used by scripts when local runtime is unavailable."""

    def __init__(
        self,
        *,
        api_url: str,
        plugin_type: str = DEFAULT_PLUGIN_TYPE,
        mode: str = DEFAULT_ANALYSIS_MODE,
        aggregation_method: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.plugin_type = plugin_type
        self.mode = mode
        self.timeout = timeout
        self.aggregation_method = aggregation_method or load_default_aggregation_method()

    def _request(self, path: Path) -> dict[str, Any]:
        files = {"file": (path.name, path.read_bytes(), "application/dicom")}
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.api_url}/api/v1/analyze",
                params={"plugin_type": self.plugin_type, "mode": self.mode},
                files=files,
            )
            response.raise_for_status()
            return response.json()

    def predict_image(self, path: Path) -> ImagePrediction:
        payload = self._request(path)
        metrics = payload.get("metrics") or {}
        runtime_loaded = bool(int(round(float(metrics.get("runtime_model_loaded", 0.0)))))
        probability = float(metrics.get("model_probability", payload.get("confidence", 0.5)))
        threshold = float(metrics.get("model_threshold", 0.5))
        disease_detected = bool(payload.get("disease_detected", False))
        if not runtime_loaded:
            disease_detected = False

        return ImagePrediction(
            path=path,
            probability=probability,
            threshold=threshold,
            disease_detected=disease_detected,
            runtime_model_loaded=runtime_loaded,
            metadata=dict(payload.get("metadata") or {}),
            message=str(payload.get("message") or ""),
            processing_time_ms=int(payload.get("processing_time_ms") or 0),
            fold_probabilities=None,
            validation_warnings=list(payload.get("validation_warnings") or []),
            raw_response=payload,
        )

    def predict_object(self, test_object: TestObject) -> ObjectPrediction:
        image_predictions = [self.predict_image(path) for path in test_object.dicom_paths]
        probabilities = [prediction.probability for prediction in image_predictions]
        probability = aggregate_probability(probabilities, method=self.aggregation_method)
        runtime_loaded = all(prediction.runtime_model_loaded for prediction in image_predictions)
        threshold = float(image_predictions[0].threshold if image_predictions else 0.5)
        warnings = [
            warning
            for prediction in image_predictions
            for warning in (prediction.validation_warnings or [])
        ]

        if runtime_loaded:
            disease_detected = probability >= threshold
            message = (
                "API-backed classifier runtime aggregated object-level prediction "
                f"using '{self.aggregation_method}'."
            )
        else:
            disease_detected = False
            message = (
                "API fallback mode was used during object-level aggregation. "
                "Result is non-diagnostic."
            )

        metadata = dict(image_predictions[0].metadata) if image_predictions else {}
        return ObjectPrediction(
            object_id=test_object.object_id,
            probability=float(probability),
            threshold=threshold,
            disease_detected=disease_detected,
            runtime_model_loaded=runtime_loaded,
            num_images_in_object=len(test_object.dicom_paths),
            aggregation_method=self.aggregation_method,
            fold_probabilities=None,
            message=message,
            metadata=metadata,
            validation_warnings=warnings,
            image_predictions=image_predictions,
        )


def save_jpeg(image: Image.Image, output_path: Path, *, quality: int = 90) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="JPEG", quality=quality, optimize=True)
