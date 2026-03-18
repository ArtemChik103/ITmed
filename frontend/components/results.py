"""Result rendering for doctor and education Streamlit modes."""
from __future__ import annotations

from typing import Any

import streamlit as st

from frontend.utils.clinical_report_builder import geometry_available, geometry_metric_rows, geometry_reason
from frontend.utils.medical_text import get_detailed_report
from frontend.utils.report_formatting import (
    compact_metrics,
    disease_color,
    disease_label,
    doctor_summary,
    education_explanations,
    keypoint_status_text,
    model_probability,
    model_threshold,
    metadata_summary,
    runtime_status_text,
)


def _render_metric_cards(result: dict[str, Any]) -> None:
    cards = compact_metrics(result)
    for start in range(0, len(cards), 3):
        row_cards = cards[start : start + 3]
        columns = st.columns(len(row_cards))
        for column, (label, value) in zip(columns, row_cards):
            column.markdown(
                f"""
                <div data-testid="stMetric" style="text-align: center; margin-bottom: 0.8rem;">
                    <div data-testid="stMetricLabel">{label}</div>
                    <div data-testid="stMetricValue" style="font-size:1.6rem!important;">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_result_header(result: dict[str, Any]) -> None:
    threshold = model_threshold(result)
    threshold_text = f"{threshold * 100:.1f}%" if threshold is not None else "не указан"
    conf_text = f"{model_probability(result) * 100:.1f}%"
    diag_label = disease_label(result)
    
    color_var = "--danger" if disease_color(result) == "#b91c1c" else "--success"
    if disease_color(result) == "#a16207":
        color_var = "--warning"
        
    glow_var = f"{color_var}-glow"

    st.markdown(
        f"""
        <div style="
            background: rgba(10, 14, 20, 0.6); 
            backdrop-filter: blur(20px); 
            border: 1px solid var({color_var}); 
            box-shadow: 0 0 20px var({glow_var});
            border-radius: 28px; 
            padding: 2rem; 
            text-align: center;
            margin-bottom: 2rem;
        ">
            <h4 style="color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.1em; margin-top: 0;">Итоговое решение</h4>
            <h1 style="color: var({color_var}); font-size: 3.5rem !important; margin: 0.5rem 0; text-shadow: 0 0 15px var({glow_var});">
                {diag_label}
            </h1>
            <p style="color: var(--text-secondary); font-size: 1.1rem; margin-bottom: 0;">
                Уверенность <strong>{conf_text}</strong> &bull; Порог <strong>{threshold_text}</strong>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_results(result: dict[str, Any], *, mode: str) -> None:
    if not result:
        st.info("После запуска анализа здесь появится клиническая сводка.")
        return

    _render_result_header(result)
    st.write("")
    _render_metric_cards(result)
    tabs = ["Сводка", "Метаданные"]
    if mode == "education":
        tabs.extend(["Пояснения", "JSON"])

    rendered_tabs = st.tabs(tabs)

    with rendered_tabs[0]:
        st.write(doctor_summary(result))
        if result.get("message"):
            st.caption(result.get("message"))
        warnings = result.get("validation_warnings") or []
        if warnings:
            st.write("Предупреждения:")
            for warning in warnings:
                st.warning(warning)

    with rendered_tabs[1]:
        metadata_cols = st.columns(2)
        summary_items = metadata_summary(result)
        for index, (label, value) in enumerate(summary_items):
            metadata_cols[index % 2].metric(label, value)

    if mode == "education":
        with rendered_tabs[2]:
            st.subheader("Геометрия")
            st.info(geometry_reason(result))
            if geometry_available(result):
                st.table(
                    [{"Показатель": label, "Значение": value} for label, value in geometry_metric_rows(result)]
                )
            st.info(keypoint_status_text(result))
            for item in education_explanations(result):
                st.write(f"- {item}")
        with rendered_tabs[3]:
            st.json(result)
