"""Streamlit UI for the classifier-first hip dysplasia demo."""
from __future__ import annotations

import inspect
import os
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

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

API_URL = os.getenv("API_URL", "http://localhost:8000")
PLUGIN_TYPE = "hip_dysplasia"

st.set_page_config(
    page_title="ИТ+Мед 2026 | DDH Demo",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
          @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&display=swap');

          /* Premium Dark Glassmorphism Aesthetics */
          :root {
            /* Core Backgrounds */
            --bg-base: #030508;
            --bg-surface: rgba(14, 18, 25, 0.45);
            --bg-surface-elevated: rgba(22, 28, 40, 0.55);
            
            /* Borders & Lines */
            --glass-border: rgba(255, 255, 255, 0.08);
            --glass-border-highlight: rgba(255, 255, 255, 0.15);
            --glass-highlight: rgba(255, 255, 255, 0.04);
            
            /* Typography Colors */
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            
            /* Accents */
            --accent-glow: rgba(0, 230, 255, 0.15);
            --accent-neon: #00e6ff;
            --accent-secondary: #ff2a6d;
            
            /* Status Colors */
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --success-glow: rgba(16, 185, 129, 0.15);
            --warning-glow: rgba(245, 158, 11, 0.15);
            --danger-glow: rgba(239, 68, 68, 0.15);

            /* Typography Settings */
            --font-display: 'Syne', sans-serif;
            --font-body: 'Outfit', sans-serif;
          }

          html, body, [class*="css"] {
            font-family: var(--font-body);
          }

          /* Global App Background with Dramatic Lighting */
          .stApp, [data-testid="stAppViewContainer"] {
            background-color: var(--bg-base);
            background-image: 
              radial-gradient(circle at 15% 10%, var(--accent-glow) 0%, transparent 40%),
              radial-gradient(circle at 85% 60%, rgba(255, 42, 109, 0.08) 0%, transparent 40%);
            background-attachment: fixed;
            color: var(--text-primary);
          }

          [data-testid="stHeader"] {
            background: transparent !important;
          }

          .block-container {
            max-width: 1540px;
            padding-top: 1rem;
            padding-bottom: 4rem;
          }

          /* Sleek Sidebar */
          [data-testid="stSidebar"] {
            background: rgba(8, 11, 15, 0.7) !important;
            backdrop-filter: blur(24px);
            -webkit-backdrop-filter: blur(24px);
            border-right: 1px solid var(--glass-border);
          }
          [data-testid="stSidebar"] * {
            color: var(--text-primary);
          }
          [data-testid="stSidebar"] .stCaption {
            color: var(--text-muted);
            font-weight: 500;
            letter-spacing: 0.05em;
            text-transform: uppercase;
          }

          /* Main App Shell / Container */
          .app-shell {
            padding: 1.5rem 2.5rem;
            border: 1px solid var(--glass-border);
            border-top: 1px solid var(--glass-border-highlight);
            border-radius: 32px;
            background: var(--bg-surface);
            backdrop-filter: blur(32px);
            -webkit-backdrop-filter: blur(32px);
            box-shadow: 
              0 32px 64px -16px rgba(0, 0, 0, 0.5),
              inset 0 1px 0 0 rgba(255, 255, 255, 0.05);
            margin-bottom: 2rem;
            position: relative;
            overflow: hidden;
          }

          /* Shell highlight effect */
          .app-shell::before {
             content: '';
             position: absolute;
             top: 0; left: 0; right: 0; height: 1px;
             background: linear-gradient(90deg, transparent, var(--accent-neon), transparent);
             opacity: 0.3;
          }

          /* Typography */
          h1, h2, h3, h4 {
            font-family: var(--font-display);
            font-weight: 700;
            letter-spacing: -0.02em;
            color: var(--text-primary);
          }
          h1 {
            font-size: 3rem !important;
            line-height: 1.1 !important;
            margin-bottom: 0.5rem !important;
            background: linear-gradient(135deg, #ffffff 0%, #94a3b8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
          }
          p, label {
            color: var(--text-secondary);
            font-size: 1.05rem;
            line-height: 1.6;
          }

          /* Interactive Elements */
          .stButton button {
            border-radius: 100px; /* Pillow shape */
            font-family: var(--font-display);
            font-weight: 600;
            letter-spacing: 0.02em;
            transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
            text-transform: uppercase;
            font-size: 0.9rem;
            padding: 0.5rem 1.5rem;
          }
          
          /* Primary Button (Neon Cyberpunk style) */
          .stButton button[kind="primary"] {
            background: transparent;
            border: 1px solid var(--accent-neon);
            color: var(--accent-neon);
            box-shadow: 0 0 15px -3px var(--accent-glow);
          }
          .stButton button[kind="primary"]:hover {
            background: var(--accent-neon);
            color: #000;
            box-shadow: 0 0 25px 5px var(--accent-glow);
            transform: translateY(-2px);
          }

          /* Secondary Button */
          .stButton button[kind="secondary"] {
            background: var(--glass-highlight);
            border: 1px solid var(--glass-border);
            color: var(--text-primary);
          }
          .stButton button[kind="secondary"]:hover {
            border-color: var(--glass-border-highlight);
            background: rgba(255, 255, 255, 0.08);
            transform: translateY(-2px);
          }

          /* Base Metrics Container */
          [data-testid="stMetric"] {
            background: var(--bg-surface-elevated);
            backdrop-filter: blur(12px);
            border: 1px solid var(--glass-border);
            border-radius: 24px;
            padding: 1.25rem;
            transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
            box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.3);
          }
          [data-testid="stMetric"]:hover {
            transform: translateY(-4px) scale(1.02);
            border-color: var(--glass-border-highlight);
            box-shadow: 0 20px 40px -10px rgba(0, 0, 0, 0.5);
          }
          [data-testid="stMetricValue"] {
            font-family: var(--font-display);
            font-weight: 700;
            color: var(--text-primary);
            font-size: 2rem !important;
          }
          [data-testid="stMetricLabel"] {
            color: var(--text-muted);
            font-family: var(--font-body);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-size: 0.85rem;
            margin-bottom: 0.25rem;
          }

          /* Form and Containers */
          div[data-testid="stForm"] {
            border: 1px solid var(--glass-border);
            border-radius: 28px;
            padding: 1.5rem;
            background: rgba(10, 14, 20, 0.4);
            backdrop-filter: blur(16px);
          }

          div[data-testid="stAlert"] {
            border-radius: 20px;
            border: 1px solid var(--glass-border);
            backdrop-filter: blur(12px);
          }

          /* Premium Tabs */
          .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: rgba(0,0,0,0.2);
            padding: 6px;
            border-radius: 100px;
            border: 1px solid var(--glass-border);
          }
          .stTabs [data-baseweb="tab"] {
            border-radius: 100px;
            background: transparent;
            color: var(--text-secondary);
            border: none;
            padding: 8px 16px;
            font-weight: 500;
            transition: all 0.3s ease;
          }
          .stTabs [data-baseweb="tab"]:hover {
            color: var(--text-primary);
          }
          .stTabs [aria-selected="true"] {
            background: rgba(255, 255, 255, 0.1) !important;
            color: var(--text-primary) !important;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
          }
          .stTabs [data-baseweb="tab-highlight"] {
            display: none; /* Hide default underline */
          }

          /* Expanders */
          div[data-testid="stExpander"] details {
            border: 1px solid var(--glass-border);
            border-radius: 20px;
            background: var(--bg-surface);
            transition: all 0.3s ease;
          }
          div[data-testid="stExpander"] details:hover {
            border-color: var(--glass-border-highlight);
          }
          div[data-testid="stExpander"] summary {
            font-family: var(--font-display);
            font-weight: 600;
          }

          /* File Uploader Target Area Customization */
          div[data-testid="stFileUploader"] section {
            background: var(--bg-surface-elevated);
            border: 1px dashed rgba(0, 230, 255, 0.3);
            border-radius: 24px;
            padding: 2rem;
            transition: all 0.3s ease;
          }
          div[data-testid="stFileUploader"] section:hover {
            border: 1px dashed var(--accent-neon);
            background: rgba(0, 230, 255, 0.05);
            box-shadow: inset 0 0 20px rgba(0, 230, 255, 0.05);
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_history_sidebar() -> None:
    history = get_history()
    st.sidebar.subheader("История")
    if st.sidebar.button("Очистить историю", use_container_width=True):
        clear_history()
        history = []

    if not history:
        st.sidebar.caption("В этой сессии анализы еще не запускались.")
        return

    for entry in history[:10]:
        runtime_text = "model" if entry.get("runtime_model_loaded") else "fallback"
        title = f"{entry['timestamp']} · {entry['filename']} · {runtime_text}"
        with st.sidebar.expander(title):
            st.write(entry.get("short_summary"))
            st.json(entry.get("json_summary", {}))


def _mode_description(mode_label: str) -> str:
    if mode_label == "Врач":
        return (
            "Краткая клиническая сводка: диагноз, confidence, threshold, metadata и предупреждения."
        )
    return (
        "Те же результаты плюс anatomy overlay, пояснения простым языком и полный JSON ответа."
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


def main() -> None:
    initialize_session_state()
    _inject_styles()

    client = ApiClient(API_URL)
    api_status = client.health()
    plugins_payload = client.list_plugins()

    with st.sidebar:
        st.title("ИТ+Мед 2026")
        st.caption("Classifier-first demo поверх текущего API и plugin runtime.")
        mode_label = st.selectbox("Режим интерфейса", list(MODE_API_VALUES))
        st.caption(f"Плагин: `{PLUGIN_TYPE}`")
        st.caption(f"API: `{API_URL}`")
        st.divider()
        render_api_status(api_status, plugins_payload)
        st.divider()
        _render_history_sidebar()

    st.markdown("<div class='app-shell'>", unsafe_allow_html=True)
    st.caption("Classifier-First Demo")
    st.title("Диагностика дисплазии тазобедренных суставов")
    st.write(
        "Интерфейс показывает только реальные поля classifier runtime: диагноз, уверенность, "
        "threshold, metadata, прозрачный статус trained model или fallback и optional anatomy layer "
        "в режиме обучения, если доступен отдельный keypoint checkpoint."
    )

    top_a, top_b, top_c = st.columns([0.95, 0.95, 1.2], gap="medium")
    top_a.metric("Режим интерфейса", mode_label)
    top_b.metric("Плагин", PLUGIN_TYPE)
    top_c.caption("Как читать экран")
    top_c.write(_mode_description(mode_label))
    st.divider()

    upload_col, helper_col = st.columns([1.4, 0.9], gap="large")

    with upload_col:
        with st.form("analysis_form", clear_on_submit=False):
            uploaded_file = render_upload_widget()
            render_file_summary(uploaded_file)
            run_btn = st.form_submit_button(
                "Запустить анализ",
                type="primary",
                disabled=not api_status.get("ok"),
            )

    with helper_col:
        st.subheader("Что будет показано")
        st.write("- Диагноз по текущему object/image input")
        st.write("- Уверенность модели и decision threshold")
        st.write("- DICOM metadata, warnings и режим работы runtime")
        if mode_label == "Обучение":
            st.info(
                "В режиме обучения по умолчанию включен anatomy overlay, если backend вернул keypoints."
            )
        else:
            st.info("В режиме врача интерфейс держит короткую и быструю клиническую сводку.")

        show_keypoints = False
        if mode_label == "Врач":
            show_keypoints = st.checkbox("Показать ориентиры", value=False)
        else:
            show_keypoints = True

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
            st.warning(f"Превью DICOM недоступно: {preview_error}")

        if run_btn:
            with st.spinner("Идет анализ снимка..."):
                try:
                    latest_result = client.analyze(
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
    left_col, right_col = st.columns([1.15, 1.0], gap="large")
    with left_col:
        st.subheader("Исходный снимок")
        _render_viewer_compat(
            preview_image,
            preview_metadata,
            result=result,
            mode=MODE_API_VALUES[mode_label],
            show_keypoints=show_keypoints,
        )

    with right_col:
        st.subheader("Результат анализа")
        render_results(result or {}, mode=MODE_API_VALUES[mode_label])
        if result:
            pdf_bytes = generate_pdf_report(result, uploaded_file.name if uploaded_file else "unknown.dcm")
            st.download_button(
                label="Скачать PDF отчет",
                data=pdf_bytes,
                file_name="report.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True
            )

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
