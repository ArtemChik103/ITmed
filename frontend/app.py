"""Streamlit UI for ИТ+Мед 2026."""
from __future__ import annotations

import os

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
TRAIN_DATA_DIR = os.getenv("TRAIN_DATA_DIR", "../train")
TEST_DATA_DIR = os.getenv("TEST_DATA_DIR", "../test_done")
PLUGIN_TYPE = "hip_dysplasia"
MODE_MAP = {"Врач": "doctor", "Обучение": "education"}

st.set_page_config(
    page_title="ИТ+Мед 2026 | AI Diagnostics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    st.title("🏥 ИТ+Мед 2026")
    st.divider()
    mode_label = st.selectbox("Режим работы", list(MODE_MAP))
    st.caption(f"Плагин: `{PLUGIN_TYPE}`")
    st.caption(f"Train data: `{TRAIN_DATA_DIR}`")
    st.caption(f"Test data: `{TEST_DATA_DIR}`")
    st.divider()

    if st.button("ℹ️ О системе"):
        st.info(
            "**ИТ+Мед 2026**\n\n"
            "Система анализа медицинских снимков (DICOM) с plugin-архитектурой.\n\n"
            f"- Режим: **{mode_label}**\n"
            f"- Плагин: **{PLUGIN_TYPE}**\n"
            f"- API: `{API_URL}`"
        )

    st.divider()
    st.caption("Статус API")
    try:
        response = httpx.get(f"{API_URL}/health", timeout=2.0)
        if response.status_code == 200:
            st.success("API работает")
        else:
            st.warning(f"API вернул статус {response.status_code}")
    except Exception:
        st.error("API недоступен")

st.title("🏥 ИТ+Мед 2026: AI Diagnostics")
st.caption(f"Режим: **{mode_label}** · API: `{API_URL}`")
st.divider()

uploaded_file = st.file_uploader(
    "Загрузите DICOM снимок",
    type=["dcm", "dicom"],
    help="Поддерживаемые форматы: .dcm, .dicom",
)

if uploaded_file is not None:
    col_info, col_btn = st.columns([3, 1])
    with col_info:
        st.success(f"Файл загружен: **{uploaded_file.name}** ({uploaded_file.size:,} байт)")
    with col_btn:
        run_btn = st.button("Запустить анализ", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner("Анализ снимка..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/octet-stream")}
                response = httpx.post(
                    f"{API_URL}/api/v1/analyze",
                    params={"plugin_type": PLUGIN_TYPE, "mode": MODE_MAP[mode_label]},
                    files=files,
                    timeout=60.0,
                )
                response.raise_for_status()
                result = response.json()

                st.divider()
                st.subheader("Результаты анализа")

                c1, c2, c3 = st.columns(3)
                c1.metric(
                    "Патология",
                    "Обнаружена" if result["disease_detected"] else "Не обнаружена",
                )
                c2.metric("Уверенность", f"{result['confidence']:.0%}")
                c3.metric("Время обработки", f"{result['processing_time_ms']} мс")

                st.info(result["message"])

                if result.get("validation_warnings"):
                    with st.expander("Предупреждения валидации"):
                        for warning in result["validation_warnings"]:
                            st.warning(warning)

                if result.get("metrics"):
                    st.subheader("Метрики")
                    metrics_cols = st.columns(len(result["metrics"]))
                    for col, (key, value) in zip(metrics_cols, result["metrics"].items()):
                        if float(value).is_integer():
                            col.metric(key, int(value))
                        else:
                            col.metric(key, f"{value:.3f}")

                with st.expander("Полный JSON ответ"):
                    st.json(result)

            except httpx.ConnectError:
                st.error("Не удалось подключиться к API. Убедитесь, что сервис запущен.")
            except httpx.HTTPStatusError as exc:
                st.error(f"Ошибка API [{exc.response.status_code}]: {exc.response.text}")
            except Exception as exc:
                st.error(f"Неожиданная ошибка: {exc}")
else:
    st.info("Загрузите .dcm файл, чтобы начать анализ.")
    st.caption(f"Внешние датасеты: `{TRAIN_DATA_DIR}`, `{TEST_DATA_DIR}`")
