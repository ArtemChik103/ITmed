"""Upload UI components."""
from __future__ import annotations

import streamlit as st


def render_upload_widget():
    st.markdown(
        """
        <div style="margin-bottom: 0.5rem; text-align: center;">
            <h3 style="margin:0; font-size: 1.5rem; color: var(--accent-neon); text-shadow: 0 0 10px rgba(0,230,255,0.3);">
                ✨ Анализ DICOM
            </h3>
            <p style="margin:0; color: var(--text-muted); font-size: 0.9rem;">Перетащите снимок в зону загрузки</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return st.file_uploader(
        "Загрузите DICOM снимок",
        type=["dcm", "dicom"],
        help="Поддерживаются только .dcm и .dicom, как и в текущем API.",
        label_visibility="collapsed",
    )


def render_file_summary(uploaded_file) -> None:
    if uploaded_file is None:
        st.info("Загрузите .dcm файл, чтобы начать анализ.")
        return

    st.caption(f"Файл: `{uploaded_file.name}` · размер: {uploaded_file.size:,} байт")
