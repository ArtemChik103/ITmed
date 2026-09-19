"""DICOM preview rendering for the Streamlit frontend."""
from __future__ import annotations

import io
from typing import Any

import numpy as np
import pydicom
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from frontend.components.keypoint_overlay import (
    render_anatomical_landmarks_overlay,
    render_keypoint_overlay,
)
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
        pixel_array = None
        try:
            from pydicom.pixel_data_handlers.util import apply_voi_lut
            pixel_array = apply_voi_lut(dataset.pixel_array, dataset).astype(np.float32)
        except Exception:
            pixel_array = dataset.pixel_array.astype(np.float32)
            if hasattr(dataset, "RescaleSlope") and hasattr(dataset, "RescaleIntercept"):
                pixel_array = float(dataset.RescaleSlope) * pixel_array + float(dataset.RescaleIntercept)

        frame = _single_frame(pixel_array)

        lower = float(np.percentile(frame, 0.5))
        upper = float(np.percentile(frame, 99.5))
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
    
    base_size = max(preview.width, preview.height)
    badge_font = _load_font(max(14, int(base_size * 0.020)))

    is_pathology = bool(result.get("disease_detected"))
    tonnis_val = int(round(float((result.get("metrics") or {}).get("tonnis_grade", 2 if is_pathology else 0))))
    conf_pct = confidence_text(model_probability(result))
    diag_name = disease_label(result)

    if not runtime_model_loaded(result):
        badge_bg = (180, 83, 9, 220)
        badge_border = (245, 158, 11, 255)
        pill_text = f"Fallback • {diag_name} ({conf_pct})"
    elif is_pathology:
        badge_bg = (153, 27, 27, 220)
        badge_border = (239, 68, 68, 255)
        pill_text = f"{diag_name} ({conf_pct}) • Тённис {tonnis_val}"
    else:
        badge_bg = (6, 95, 70, 220)
        badge_border = (16, 185, 129, 255)
        pill_text = f"{diag_name} ({conf_pct}) • Тённис 0"

    bbox_pill = draw.textbbox((0, 0), pill_text, font=badge_font)
    pw = bbox_pill[2] - bbox_pill[0]
    ph = bbox_pill[3] - bbox_pill[1]
    p_pad_x = max(12, int(base_size * 0.014))
    p_pad_y = max(6, int(base_size * 0.008))
    pos_x = int(base_size * 0.025)
    pos_y = int(base_size * 0.025)

    draw.rounded_rectangle(
        [pos_x, pos_y, pos_x + pw + p_pad_x * 2, pos_y + ph + p_pad_y * 2],
        radius=max(6, int(base_size * 0.008)),
        fill=badge_bg,
        outline=badge_border,
        width=2,
    )
    draw.text((pos_x + p_pad_x, pos_y + p_pad_y - 2), pill_text, font=badge_font, fill=(255, 255, 255, 255))

    rendered = canvas.convert("RGB")
    keypoints = (result.get("keypoints") or []) if result else []
    if show_keypoints:
        if keypoints:
            rendered = render_keypoint_overlay(
                rendered,
                keypoints,
                labels=overlay_keypoint_labels(),
            )
        else:
            rendered = render_anatomical_landmarks_overlay(
                rendered,
                result=result,
                keypoints=keypoints,
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

    # Windowing / Contrast Controls
    with st.expander("Регулировка яркости и контрастности (Windowing W/L)", expanded=False):
        c_p1, c_p2 = st.columns(2)
        with c_p1:
            wl_preset = st.selectbox(
                "Пресет окна визуализации",
                ["Стандартное (DICOM VOI LUT)", "Высокий контраст (Кости)", "Мягкие ткани", "Инверсия рентгенограммы"],
                key="wl_preset_select",
            )
        with c_p2:
            st.caption("Быстрая настройка динамического диапазона для оценки костных структур.")

        s_col1, s_col2 = st.columns(2)
        with s_col1:
            brightness_val = st.slider("Яркость (Level)", -50, 50, 0, step=5, key="wl_slider_level")
        with s_col2:
            contrast_val = st.slider("Контрастность (Width)", 50, 200, 100, step=10, key="wl_slider_width")

    # Apply W/L presets and adjustments to preview image
    adjusted_preview = preview.copy()
    if wl_preset == "Инверсия рентгенограммы":
        from PIL import ImageOps
        adjusted_preview = ImageOps.invert(adjusted_preview.convert("L"))
    elif wl_preset == "Высокий контраст (Кости)":
        from PIL import ImageEnhance
        adjusted_preview = ImageEnhance.Contrast(adjusted_preview).enhance(1.45)
    elif wl_preset == "Мягкие ткани":
        from PIL import ImageEnhance
        adjusted_preview = ImageEnhance.Contrast(adjusted_preview).enhance(0.75)
        adjusted_preview = ImageEnhance.Brightness(adjusted_preview).enhance(1.15)

    if contrast_val != 100:
        from PIL import ImageEnhance
        adjusted_preview = ImageEnhance.Contrast(adjusted_preview).enhance(contrast_val / 100.0)
    if brightness_val != 0:
        from PIL import ImageEnhance
        adjusted_preview = ImageEnhance.Brightness(adjusted_preview).enhance(1.0 + brightness_val / 100.0)

    overlay_img = build_overlay_image(adjusted_preview, result, show_keypoints=show_keypoints)
    st.image(overlay_img, use_container_width=True)

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
    st.markdown(f"<div style='margin-top:0.75rem;margin-bottom:0.75rem;'>{chips_html}</div>", unsafe_allow_html=True)

    # Download high-res annotated PNG & DICOM Secondary Capture
    buf = io.BytesIO()
    overlay_img.save(buf, format="PNG")
    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button(
            label="Скачать снимок (PNG)",
            data=buf.getvalue(),
            file_name=f"annotated_pelvis_{preview_metadata.get('modality', 'DX')}.png",
            mime="image/png",
            use_container_width=True,
        )
    with dl_col2:
        try:
            from frontend.utils.dicom_export import create_dicom_secondary_capture

            dcm_bytes = create_dicom_secondary_capture(
                overlay_img,
                metadata=preview_metadata,
                clinical_result=result,
            )
            st.download_button(
                label="Экспорт DICOM SC (.dcm)",
                data=dcm_bytes,
                file_name=f"secondary_capture_{preview_metadata.get('modality', 'DX')}.dcm",
                mime="application/dicom",
                use_container_width=True,
            )
        except Exception as exc:
            st.caption(f"Экспорт DICOM SC: {exc}")

    if result is None or mode != "education":
        return

    if keypoint_model_loaded(result) and keypoint_count(result) > 0:
        st.caption(keypoint_status_text(result))
    else:
        st.info(keypoint_status_text(result))
