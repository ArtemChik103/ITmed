"""
frontend/app.py — Streamlit UI for ИТ+Мед 2026.
Phase 1: file upload + API call + results display.
"""
import os

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ИТ+Мед 2026 | AI Diagnostics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏥 ИТ+Мед 2026")
    st.divider()
    mode = st.selectbox("Режим работы", ["Врач", "Обучение"])
    st.divider()
    if st.button("ℹ️ О системе"):
        st.info(
            "**ИТ+Мед 2026**\n\n"
            "Система анализа медицинских снимков (DICOM) с ИИ.\n\n"
            f"- Режим: **{mode}**\n"
            "- Версия: **1.0.0** (Phase 1)\n"
            f"- API: `{API_URL}`"
        )
    # API status
    st.divider()
    st.caption("Статус API")
    try:
        r = httpx.get(f"{API_URL}/health", timeout=2.0)
        if r.status_code == 200:
            st.success("🟢 API работает")
        else:
            st.warning(f"🟡 API вернул статус {r.status_code}")
    except Exception:
        st.error("🔴 API недоступен")

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("🏥 ИТ+Мед 2026: AI Diagnostics")
st.caption(f"Режим: **{mode}** · API: `{API_URL}`")
st.divider()

uploaded_file = st.file_uploader(
    "📂 Загрузите DICOM снимок",
    type=["dcm", "dicom"],
    help="Поддерживаемые форматы: .dcm, .dicom",
)

if uploaded_file is not None:
    col_info, col_btn = st.columns([3, 1])
    with col_info:
        st.success(f"✅ Файл загружен: **{uploaded_file.name}** ({uploaded_file.size:,} байт)")
    with col_btn:
        run_btn = st.button("🚀 Запустить анализ", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner("⏳ Анализ снимка..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/octet-stream")}
                response = httpx.post(
                    f"{API_URL}/api/v1/analyze",
                    files=files,
                    timeout=60.0,
                )
                response.raise_for_status()
                result = response.json()

                # ── Results ────────────────────────────────────────────────
                st.divider()
                st.subheader("📊 Результаты анализа")

                c1, c2, c3 = st.columns(3)
                c1.metric(
                    "Патология",
                    "⚠️ Обнаружена" if result["disease_detected"] else "✅ Не обнаружена",
                )
                c2.metric("Уверенность модели", f"{result['confidence']:.0%}")
                c3.metric("Время обработки", f"{result['processing_time_ms']} мс")

                st.info(f"💬 {result['message']}")

                if result.get("metrics"):
                    st.subheader("📈 Метрики")
                    metrics_cols = st.columns(len(result["metrics"]))
                    for col, (k, v) in zip(metrics_cols, result["metrics"].items()):
                        col.metric(k.capitalize(), f"{v:.3f}")

                with st.expander("🔍 Полный JSON ответ от API"):
                    st.json(result)

            except httpx.ConnectError:
                st.error("❌ Не удалось подключиться к API. Убедитесь, что сервис запущен.")
            except httpx.HTTPStatusError as e:
                st.error(f"❌ Ошибка API [{e.response.status_code}]: {e.response.text}")
            except Exception as e:
                st.error(f"❌ Неожиданная ошибка: {e}")
else:
    st.info("👆 Загрузите .dcm файл чтобы начать анализ.")
    st.caption(f"Датасет: `train/Норма/`, `train/Патология/`, `test_done/`")
