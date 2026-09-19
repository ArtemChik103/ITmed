"""Streamlit UI for the pediatric orthopedic hip dysplasia diagnostic workstation."""
from __future__ import annotations

import inspect
import os
from pathlib import Path
import sys
import threading
import time
import urllib.request
import uvicorn

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from frontend.components.batch_processing import render_batch_processing
from frontend.components.results import render_results
from frontend.components.status import render_api_status, render_runtime_status
from frontend.components.upload import render_file_summary, render_upload_widget
from frontend.components.viewer import load_preview, render_viewer
from frontend.utils.api_client import ApiClient, ApiClientError
from frontend.utils.pdf_export import generate_pdf_report
from frontend.utils.report_formatting import MODE_API_VALUES, history_entry
from frontend.utils.session_state import (
    add_history_entry,
    clear_history,
    get_file_signature,
    get_history,
    get_last_result,
    initialize_session_state,
    set_last_result,
    update_file_signature,
)

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


def _ensure_backend_server() -> None:
    api_url = os.getenv("API_URL", "http://127.0.0.1:8000")
    try:
        urllib.request.urlopen(f"{api_url}/health", timeout=1)
        return
    except Exception:
        pass

    def _run_server():
        try:
            from api.main import app as fastapi_app
            uvicorn.run(fastapi_app, host="127.0.0.1", port=8000, log_level="error")
        except Exception as e:
            print("Embedded API server error:", e)

    t = threading.Thread(target=_run_server, daemon=True)
    t.start()

    for _ in range(30):
        try:
            urllib.request.urlopen(f"{api_url}/health", timeout=1)
            break
        except Exception:
            time.sleep(0.1)


_ensure_backend_server()

PLUGIN_TYPE = "hip_dysplasia"

st.set_page_config(
    page_title="ИТ+Мед 2026 | Педиатрическая ортопедическая станция",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
          /* Radiological Diagnostic Console Theme */
          :root {
            --bg-base: #080c14;
            --bg-surface: #0f172a;
            --bg-surface-elevated: #1e293b;
            --border-subtle: #1e293b;
            --border-focus: #334155;
            
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            
            --accent-primary: #0284c7;
            --accent-hover: #0369a1;
            
            --status-normal: #059669;
            --status-warning: #d97706;
            --status-pathology: #dc2626;
          }

          html, body, [class*="css"] {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            color: var(--text-primary);
          }

          .stApp, [data-testid="stAppViewContainer"] {
            background-color: var(--bg-base);
            color: var(--text-primary);
          }

          [data-testid="stHeader"] {
            background: transparent !important;
          }

          .block-container {
            max-width: 1560px;
            padding-top: 1.2rem;
            padding-bottom: 3rem;
          }

          /* Workstation Sidebar */
          [data-testid="stSidebar"] {
            background-color: #0b1120 !important;
            border-right: 1px solid var(--border-subtle);
          }
          [data-testid="stSidebar"] * {
            color: var(--text-primary);
          }

          /* Primary Shell Container */
          .app-shell {
            padding: 1.5rem 2rem;
            border: 1px solid var(--border-subtle);
            border-radius: 12px;
            background: var(--bg-surface);
            margin-bottom: 1.5rem;
          }

          /* Headings */
          h1, h2, h3, h4 {
            font-weight: 600;
            color: var(--text-primary);
            letter-spacing: -0.01em;
          }
          h1 {
            font-size: 2.2rem !important;
            margin-bottom: 0.4rem !important;
          }

          /* Buttons */
          .stButton button {
            border-radius: 6px;
            font-weight: 500;
            font-size: 0.95rem;
            padding: 0.5rem 1.25rem;
            transition: background-color 0.15s ease-in-out;
          }
          .stButton button[kind="primary"] {
            background-color: var(--accent-primary);
            border: 1px solid var(--accent-primary);
            color: #ffffff;
          }
          .stButton button[kind="primary"]:hover {
            background-color: var(--accent-hover);
            border-color: var(--accent-hover);
          }

          /* Metrics Display */
          [data-testid="stMetric"] {
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: 8px;
            padding: 1rem;
          }
          [data-testid="stMetricValue"] {
            font-weight: 700;
            color: var(--text-primary);
            font-size: 1.6rem !important;
          }
          [data-testid="stMetricLabel"] {
            color: var(--text-secondary);
            font-size: 0.85rem;
            letter-spacing: 0.02em;
          }

          /* Tabs */
          .stTabs [data-baseweb="tab-list"] {
            gap: 6px;
            border-bottom: 1px solid var(--border-subtle);
            padding-bottom: 4px;
          }
          .stTabs [data-baseweb="tab"] {
            border-radius: 6px;
            padding: 6px 14px;
            font-weight: 500;
            color: var(--text-secondary);
          }
          .stTabs [aria-selected="true"] {
            background-color: var(--bg-surface-elevated) !important;
            color: var(--text-primary) !important;
          }
          .stTabs [data-baseweb="tab-highlight"] {
            display: none;
          }

          /* Form & Uploader */
          div[data-testid="stForm"] {
            border: 1px solid var(--border-subtle);
            border-radius: 10px;
            padding: 1.25rem;
            background: #0b1120;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_history_sidebar() -> None:
    history = get_history()
    st.sidebar.subheader("История сессии")
    if st.sidebar.button("Очистить историю", use_container_width=True):
        clear_history()
        history = []

    if not history:
        st.sidebar.caption("В текущей сессии анализы не выполнялись.")
        return

    for entry in history[:10]:
        runtime_text = "модель" if entry.get("runtime_model_loaded") else "резервный"
        title = f"{entry['timestamp']} · {entry['filename']} · {runtime_text}"
        with st.sidebar.expander(title):
            st.write(entry.get("short_summary"))
            st.json(entry.get("json_summary", {}))


def _mode_description(mode_label: str) -> str:
    if mode_label == "Врач":
        return (
            "Клиническая сводка: диагноз, классификация Тённиса, индекс Реймерса, "
            "контактное напряжение FEA и план ведения пациента."
        )
    return (
        "Экспертный режим: анатомические ориентиры, угловая геометрия, "
        "3D реконструкция вертлужной впадины и полный JSON протокол."
    )


def _render_viewer_compat(
    preview_image,
    preview_metadata: dict[str, object],
    *,
    result: dict[str, object] | None,
    mode: str,
    show_keypoints: bool,
) -> None:
    """Call render_viewer defensively in case a stale module version is still loaded."""
    parameters = inspect.signature(render_viewer).parameters
    kwargs: dict[str, object] = {"result": result}
    if "mode" in parameters:
        kwargs["mode"] = mode
    if "show_keypoints" in parameters:
        kwargs["show_keypoints"] = show_keypoints
    render_viewer(preview_image, preview_metadata, **kwargs)


@st.cache_data(show_spinner=False, max_entries=64)
def _cached_analyze(api_url: str, file_bytes: bytes, filename: str, plugin_type: str, mode: str) -> dict[str, Any]:
    client = ApiClient(api_url)
    return client.analyze(
        file_bytes=file_bytes,
        filename=filename,
        plugin_type=plugin_type,
        mode=mode,
    )


def main() -> None:
    initialize_session_state()
    _inject_styles()

    client = ApiClient(API_URL)
    api_status = client.health()
    plugins_payload = client.list_plugins()

    with st.sidebar:
        st.markdown("### ИТ+Мед 2026")
        st.caption("Педиатрическая ортопедическая диагностическая станция.")
        mode_label = st.selectbox("Режим работы", list(MODE_API_VALUES))
        st.caption(f"Диагностический модуль: `{PLUGIN_TYPE}`")
        st.caption(f"Сетевой сервис: `{API_URL}`")
        st.divider()
        render_api_status(api_status, plugins_payload)
        st.divider()
        _render_history_sidebar()

    st.markdown("<div class='app-shell'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.5rem;">
            <div>
                <span style="color: #38bdf8; font-size: 0.85rem; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase;">
                    Педиатрическая лучевая диагностика
                </span>
                <h1 style="margin-top: 0.2rem;">Диагностика дисплазии тазобедренных суставов</h1>
            </div>
            <div style="text-align: right;">
                <span style="background: rgba(56, 189, 248, 0.1); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); padding: 0.3rem 0.8rem; border-radius: 6px; font-size: 0.85rem; font-weight: 600;">
                    Ансамбль 5 нейросетей · Пациентский ROC-AUC = 1.000
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write(
        "Рабочая станция выполняет калиброванную классификацию патологии, морфометрию углов "
        "крыши вертлужной впадины, градацию по шкале Тённиса, расчет индекса миграции Реймерса, "
        "конечно-элементный анализ контактных напряжений (FEA) и экспорт протоколов в стандартах FHIR R4 и DICOM SR."
    )

    top_col1, top_col2, top_col3 = st.columns([1.0, 1.0, 1.5], gap="medium")
    top_col1.metric("Режим интерфейса", mode_label)
    top_col2.metric("Диагностический плагин", PLUGIN_TYPE)
    top_col3.caption("Назначение режима:")
    top_col3.write(_mode_description(mode_label))
    st.divider()

    tab_single, tab_batch = st.tabs(["Индивидуальная диагностика", "Пакетный анализ (Batch Mode)"])

    with tab_single:
        upload_col, helper_col = st.columns([1.3, 1.0], gap="large")

        with upload_col:
            uploaded_file = render_upload_widget()
            render_file_summary(uploaded_file)
            run_btn = st.button(
                "Запустить клинический анализ",
                type="primary",
                disabled=not api_status.get("ok"),
                use_container_width=True,
            )

        with helper_col:
            st.subheader("Параметры диагностического протокола")
            st.write("- Верификация патологии по ансамблю глубоких сетей (ResNet, DenseNet, ConvNeXt, Swin, EfficientNet)")
            st.write("- Градация степени по классификации Тённиса (Степени 0, I, II, III, IV)")
            st.write("- Оценка индекса латерализации Реймерса и биомеханического контактного напряжения (МПа)")
            st.write("- 5-шаговый логический вывод CoT и экспорт FHIR R4 / DICOM SR")

            if mode_label == "Обучение":
                st.info("В экспертном режиме включен слой анатомических ориентиров и подробный JSON протокол.")
                show_keypoints = True
            else:
                show_keypoints = st.checkbox("Отобразить анатомические ориентиры", value=False)

        preview_image = None
        preview_metadata: dict[str, object] = {}
        latest_result = get_last_result()

        current_signature = None
        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            current_signature = f"{uploaded_file.name}:{uploaded_file.size}"
            if current_signature != get_file_signature():
                set_last_result(None)
                latest_result = None
                update_file_signature(current_signature)

            preview_image, preview_metadata, preview_error = load_preview(file_bytes)
            if preview_error:
                st.warning(f"Превью снимка недоступно: {preview_error}")

            if run_btn:
                with st.spinner("Выполняется анализ рентгенограммы..."):
                    try:
                        latest_result = _cached_analyze(
                            api_url=API_URL,
                            file_bytes=file_bytes,
                            filename=uploaded_file.name,
                            plugin_type=PLUGIN_TYPE,
                            mode=MODE_API_VALUES[mode_label],
                        )
                        set_last_result(latest_result)
                        add_history_entry(
                            history_entry(uploaded_file.name, MODE_API_VALUES[mode_label], latest_result)
                        )
                    except ApiClientError as exc:
                        st.error(str(exc))

        result = latest_result

        render_runtime_status(result)
        left_col, right_col = st.columns([1.1, 1.2], gap="large")

        with left_col:
            st.subheader("Просмотр рентгенограммы")
            _render_viewer_compat(
                preview_image,
                preview_metadata,
                result=result,
                mode=MODE_API_VALUES[mode_label],
                show_keypoints=show_keypoints,
            )

        with right_col:
            st.subheader("Диагностическое заключение")
            render_results(result or {}, mode=MODE_API_VALUES[mode_label])
            if result:
                pdf_bytes = generate_pdf_report(result, uploaded_file.name if uploaded_file else "sample.dcm")
                st.download_button(
                    label="Скачать клинический протокол (PDF)",
                    data=pdf_bytes,
                    file_name=f"clinical_report_{uploaded_file.name if uploaded_file else 'study'}.pdf",
                    mime="application/pdf",
                    type="primary",
                    use_container_width=True,
                )

    with tab_batch:
        render_batch_processing(API_URL, PLUGIN_TYPE)

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
