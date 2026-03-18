"""DICOM preview rendering for the Streamlit frontend."""
from __future__ import annotations

import io
from typing import Any

import numpy as np
import pydicom
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from frontend.components.keypoint_overlay import render_keypoint_overlay
from frontend.utils.keypoint_labels import overlay_keypoint_labels
from frontend.utils.report_formatting import (
    confidence_text,
    disease_label,
    keypoint_count,
    keypoint_model_loaded,
    keypoint_status_text,
    model_probability,
    model_threshold,
    runtime_model_loaded,
)


def _load_font(size: int):
    for font_name in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _single_frame(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        return array.astype(np.float32, copy=False)
    if array.ndim == 3:
        if array.shape[0] == 1:
            return array[0].astype(np.float32, copy=False)
        if array.shape[-1] == 1:
            return array[..., 0].astype(np.float32, copy=False)
        if array.shape[-1] in {3, 4}:
            return np.mean(array[..., :3], axis=2, dtype=np.float32)
        return array[array.shape[0] // 2].astype(np.float32, copy=False)
    raise ValueError(f"Неподдерживаемая форма изображения: {list(array.shape)}")


def load_preview(file_bytes: bytes) -> tuple[Image.Image | None, dict[str, Any], str | None]:
    try:
        dataset = pydicom.dcmread(io.BytesIO(file_bytes))
        pixel_array = dataset.pixel_array.astype(np.float32)
        if hasattr(dataset, "RescaleSlope") and hasattr(dataset, "RescaleIntercept"):
            pixel_array = float(dataset.RescaleSlope) * pixel_array + float(dataset.RescaleIntercept)
        frame = _single_frame(pixel_array)

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

        if str(getattr(dataset, "PhotometricInterpretation", "")) == "MONOCHROME1":
            normalized = 1.0 - normalized

        preview = Image.fromarray(np.clip(normalized * 255.0, 0.0, 255.0).astype(np.uint8), mode="L")
        metadata = {
            "modality": str(getattr(dataset, "Modality", "") or "не указана"),
            "rows": int(getattr(dataset, "Rows", 0) or 0),
            "columns": int(getattr(dataset, "Columns", 0) or 0),
            "frames": int(getattr(dataset, "NumberOfFrames", 1) or 1),
            "photometric_interpretation": str(
                getattr(dataset, "PhotometricInterpretation", "") or "не указана"
            ),
        }
        return preview, metadata, None
    except Exception as exc:  # noqa: BLE001
        return None, {}, str(exc)


def build_overlay_image(
    preview: Image.Image,
    result: dict[str, Any] | None,
    *,
    show_keypoints: bool = False,
) -> Image.Image:
    canvas = preview.convert("RGBA")
    if result is None:
        return canvas.convert("RGB")

    draw = ImageDraw.Draw(canvas)
    
    # Sleek translucent glassmorphism banner
    # Green for ok, Red for disease, Yellow/Orange for fallback
    if not runtime_model_loaded(result):
        banner_color = (245, 158, 11, 190)  # Warning Amber
        glow_color = (252, 211, 77, 60)
    elif result.get("disease_detected"):
        banner_color = (225, 29, 72, 190)   # Rose/Red
        glow_color = (244, 63, 94, 60)
    else:
        banner_color = (16, 185, 129, 190)  # Emerald Green
        glow_color = (52, 211, 153, 60)

    base_size = max(preview.width, preview.height)
    header_size = max(28, int(base_size * 0.035))
    body_size = max(16, int(base_size * 0.02))
    
    header_font = _load_font(header_size)
    body_font = _load_font(body_size)
    
    padding = int(base_size * 0.02)
    radius = int(base_size * 0.02)
    
    box_w = max(360, int(base_size * 0.45))
    box_h = int(header_size * 1.5 + body_size * 3 + padding * 2)
    
    # Outer Glow / Soft shadow
    draw.rounded_rectangle((padding, padding, padding + box_w, padding + box_h), radius=radius, fill=glow_color)
    
    # Inner glass banner
    inner_pad = max(4, int(base_size * 0.005))
    draw.rounded_rectangle((padding + inner_pad, padding + inner_pad, padding + box_w - inner_pad, padding + box_h - inner_pad), radius=max(2, radius-inner_pad), fill=banner_color)
    
    # Neon Accent Line
    acc_w = max(8, int(base_size * 0.01))
    draw.rounded_rectangle((padding + inner_pad, padding + inner_pad, padding + inner_pad + acc_w, padding + box_h - inner_pad), radius=max(2, radius-inner_pad), fill=(255, 255, 255, 200))

    # Text content
    text_x = padding + inner_pad + acc_w + padding
    text_y = padding + inner_pad + padding
    draw.text((text_x, text_y), disease_label(result).upper(), font=header_font, fill=(255, 255, 255, 255))
    
    text_y += int(header_size * 1.5)
    draw.text(
        (text_x, text_y),
        f"Уверенность: {confidence_text(model_probability(result))}",
        font=body_font,
        fill=(255, 255, 255, 230),
    )
    
    threshold = model_threshold(result)
    if threshold is not None:
        text_y += int(body_size * 1.5)
        draw.text(
            (text_x, text_y),
            f"Порог решения: {confidence_text(threshold)}",
            font=body_font,
            fill=(255, 255, 255, 200),
        )

    rendered = canvas.convert("RGB")
    keypoints = result.get("keypoints") or []
    if show_keypoints and keypoints:
        rendered = render_keypoint_overlay(
            rendered,
            keypoints,
            labels=overlay_keypoint_labels(),
        )
    return rendered


def render_viewer(
    preview: Image.Image | None,
    preview_metadata: dict[str, Any],
    *,
    result: dict[str, Any] | None = None,
    mode: str = "doctor",
    show_keypoints: bool = False,
) -> None:
    if preview is None:
        st.markdown(
            (
                "<div style='padding:18px;border-radius:18px;border:1px solid rgba(255,255,255,0.08);"
                "background:rgba(255,255,255,0.03);color:#9ea8b7'>"
                "Превью снимка появится после загрузки DICOM."
                "</div>"
            ),
            unsafe_allow_html=True,
        )
        return

    st.image(build_overlay_image(preview, result, show_keypoints=show_keypoints), use_container_width=True)
    chips = [
        f"Modality: {preview_metadata.get('modality', 'не указана')}",
        f"Размер: {preview_metadata.get('rows', 0)}x{preview_metadata.get('columns', 0)}",
        f"Frames: {preview_metadata.get('frames', 1)}",
        f"PI: {preview_metadata.get('photometric_interpretation', 'не указана')}",
    ]
    chips_html = "".join(
        (
            "<span style='display:inline-block;margin:0 8px 8px 0;padding:6px 14px;border-radius:100px;"
            "border:1px solid rgba(255,255,255,0.08);background:rgba(255,255,255,0.02);backdrop-filter:blur(10px);"
            "color:var(--text-secondary);font-size:0.8rem;letter-spacing:0.02em;box-shadow:0 4px 10px rgba(0,0,0,0.1);'>"
            f"{chip}</span>"
        )
        for chip in chips
    )
    st.markdown(f"<div style='margin-top:1rem;'>{chips_html}</div>", unsafe_allow_html=True)

    if result is None or mode != "education":
        return

    if keypoint_model_loaded(result) and keypoint_count(result) > 0:
        st.caption(keypoint_status_text(result))
    else:
        st.info(keypoint_status_text(result))
