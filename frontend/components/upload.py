"""Upload and verified demo case selection components for DICOM images."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = REPO_ROOT / "frontend" / "data" / "demo_cases"

DEMO_CASES = {
    "Контрольная норма (Норма, Степень 0)": DEMO_DIR / "normal_control" / "normal_control.dcm",
    "Дисплазия крыши вертлужной впадины (1OGQ64, Степень 1)": DEMO_DIR / "1OGQ64" / "00000001.dcm",
    "Выраженная дисплазия тазобедренных суставов (71dphp, Степень 2)": DEMO_DIR / "71dphp" / "00000001.dcm",
    "Пограничный клинический случай (28v1xk, Контроль)": DEMO_DIR / "28v1xk" / "28v1xk.dcm",
}


class SyntheticUploadedFile:
    """Mock Streamlit UploadedFile interface for local demo files."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.name = path.name
        self._content = path.read_bytes()
        self.size = len(self._content)

    def getvalue(self) -> bytes:
        return self._content

    def read(self) -> bytes:
        return self._content


def render_upload_widget() -> Any:
    st.markdown(
        """
        <div style="margin-bottom: 0.75rem;">
            <div style="font-size: 1.1rem; font-weight: 600; color: #f1f5f9; margin-bottom: 0.25rem;">
                Источник рентгенограммы
            </div>
            <div style="color: #94a3b8; font-size: 0.85rem;">
                Загрузите файл DICOM или выберите верифицированный клинический пример
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    source_mode = st.radio(
        "Источник снимка",
        ["Клинические примеры", "Загрузка своего файла"],
        horizontal=True,
        label_visibility="collapsed",
        key="source_mode_radio",
    )

    if source_mode == "Клинические примеры":
        selected_demo = st.selectbox(
            "Выберите верифицированный клинический снимок",
            list(DEMO_CASES.keys()),
            help="Верифицированные клинические рентгенограммы из базы данных с подтвержденными диагнозами.",
            key="demo_case_selectbox",
        )
        demo_path = DEMO_CASES[selected_demo]
        if demo_path.exists():
            return SyntheticUploadedFile(demo_path)
        st.warning(f"Файл примера не найден: {demo_path}")
        return None

    return st.file_uploader(
        "Загрузите DICOM снимок",
        type=["dcm", "dicom"],
        help="Поддерживаются файлы в формате .dcm и .dicom.",
        label_visibility="collapsed",
        key="custom_dicom_uploader",
    )


def render_file_summary(uploaded_file: Any) -> None:
    if uploaded_file is None:
        st.info("Выберите файл или клинический пример для начала анализа.")
        return

    st.caption(f"Файл: `{uploaded_file.name}` | Размер: {uploaded_file.size:,} байт")
