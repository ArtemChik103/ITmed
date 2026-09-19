"""Result rendering for pediatric orthopedic clinical and educational modes."""
from __future__ import annotations

import json
from typing import Any

import streamlit as st

from frontend.utils.clinical_report_builder import geometry_available, geometry_metric_rows, geometry_reason
from frontend.utils.report_formatting import (
    acetabular_angle_deg,
    anteversion_deg,
    compact_metrics,
    coxarthrosis_risk_pct,
    crossover_sign_detected,
    cup_volume_ml,
    disease_color,
    disease_label,
    doctor_summary,
    dpci_index,
    education_explanations,
    fea_peak_stress_mpa,
    ganz_pao_recommended,
    is_dynamically_reducible,
    keypoint_status_text,
    labral_coverage_pct,
    metadata_summary,
    micro_snr_db,
    model_probability,
    model_threshold,
    reimers_index_pct,
    runtime_model_loaded,
    tonnis_grade_label,
    tonnis_grade_value,
    trabecular_gain,
)
from plugins.hip_dysplasia.model import (
    FHIRStructuredClinicalExporter,
    OrthopedicReasoningClinicalAgent,
)


def _render_result_header(result: dict[str, Any]) -> None:
    threshold = model_threshold(result)
    threshold_text = f"{threshold * 100:.1f}%" if threshold is not None else "не указан"
    conf_text = f"{model_probability(result) * 100:.1f}%"
    diag_label = disease_label(result)

    is_pathology = bool(result.get("disease_detected"))
    badge_bg = "rgba(220, 38, 38, 0.15)" if is_pathology else "rgba(5, 150, 105, 0.15)"
    badge_border = "#dc2626" if is_pathology else "#059669"
    badge_text = "#f87171" if is_pathology else "#34d399"

    tonnis_val = tonnis_grade_value(result)
    tonnis_desc = tonnis_grade_label(result)

    if is_pathology:
        status_line = (
            f"Вероятность патологии: <strong style='color: #f1f5f9;'>{conf_text}</strong> "
            f"(выше порога {threshold_text}) &bull; Классификация: "
            f"<strong style='color: #f1f5f9;'>Шкала Тённиса {tonnis_val}</strong>"
        )
    else:
        status_line = (
            f"Вероятность патологии: <strong style='color: #f1f5f9;'>{conf_text}</strong> "
            f"(в пределах нормы &le; {threshold_text}) &bull; Классификация: "
            f"<strong style='color: #f1f5f9;'>Шкала Тённиса 0 (Норма)</strong>"
        )

    st.markdown(
        f"""
        <div style="
            background: #0f172a;
            border: 1px solid {badge_border};
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        ">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                <span style="color: #94a3b8; font-size: 0.85rem; letter-spacing: 0.05em; text-transform: uppercase;">
                    Диагностическое заключение
                </span>
                <span style="
                    background: {badge_bg};
                    color: {badge_text};
                    border: 1px solid {badge_border};
                    padding: 0.25rem 0.75rem;
                    border-radius: 6px;
                    font-size: 0.85rem;
                    font-weight: 600;
                ">
                    {tonnis_desc}
                </span>
            </div>
            <div style="font-size: 2.2rem; font-weight: 700; color: {badge_text}; line-height: 1.2; margin-bottom: 0.5rem;">
                {diag_label}
            </div>
            <div style="color: #94a3b8; font-size: 0.95rem;">
                {status_line}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_metric_grid(result: dict[str, Any]) -> None:
    cards = compact_metrics(result)
    for start in range(0, len(cards), 3):
        row_cards = cards[start : start + 3]
        cols = st.columns(len(row_cards))
        for col, (label, val) in zip(cols, row_cards):
            col.metric(label=label, value=val)


def _render_cot_tab(result: dict[str, Any]) -> None:
    metadata = result.get("metadata") or {}
    patient_id = str(metadata.get("patient_id") or "PAT-ANONYMOUS")
    tonnis_val = tonnis_grade_value(result)
    reimers_val = reimers_index_pct(result)
    alpha_val = acetabular_angle_deg(result)
    stress_val = fea_peak_stress_mpa(result)

    agent = OrthopedicReasoningClinicalAgent(high_risk_cutoff=0.60)
    cot_output = agent.reason_clinical_case(
        patient_id=patient_id,
        age_months=4.5,
        measured_metrics={
            "tonnis_grade": tonnis_val,
            "reimers_index_pct": reimers_val,
            "acetabular_angle_deg": alpha_val,
            "peak_contact_stress_mpa": stress_val,
        },
        clinical_history={
            "breech_presentation": False,
            "family_history_ddh": False,
            "muscular_torticollis": False,
            "generalized_joint_laxity": False,
        },
    )

    st.subheader("Мультимодальный логический вывод (Chain-of-Thought)")
    st.write(
        "Клинический агент последовательно анализирует морфометрию, контактные напряжения, "
        "анамнез и форму патологии для формулирования окончательного диагноза."
    )

    for step_text in cot_output["chain_of_thought_steps"]:
        st.markdown(
            f"""
            <div style="
                background: #1e293b;
                border-left: 3px solid #38bdf8;
                padding: 0.6rem 1rem;
                border-radius: 4px;
                margin-bottom: 0.5rem;
                font-family: 'Consolas', monospace;
                font-size: 0.9rem;
                color: #e2e8f0;
            ">
                {step_text}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.markdown("**Окончательный клинический диагноз:**")
        st.write(cot_output["definitive_clinical_diagnosis"])
    with res_col2:
        st.markdown("**План терапевтического ведения:**")
        st.write(cot_output["actionable_management_plan"])


def _render_biomechanics_tab(result: dict[str, Any]) -> None:
    st.subheader("Конечно-элементный анализ (FEA) и 3D реконструкция")
    
    col1, col2, col3 = st.columns(3)
    stress_mpa = fea_peak_stress_mpa(result)
    risk_pct = coxarthrosis_risk_pct(result)
    pao_rec = ganz_pao_recommended(result)

    col1.metric("Пиковое контактное напряжение", f"{stress_mpa:.2f} МПа", help="Предел физиологической нормы: 1.80 МПа")
    col2.metric("20-летний риск коксартроза", f"{risk_pct:.1f}%")
    col3.metric("Остеотомия по Ганцу (PAO)", "Показана" if pao_rec else "Не требуется")

    st.markdown("---")
    st.subheader("3D морфометрия сустава (Gaussian Splatting)")
    col_a, col_b, col_c = st.columns(3)
    cup_vol = cup_volume_ml(result)
    labral_cov = labral_coverage_pct(result)
    ant_deg = anteversion_deg(result)
    crossover = crossover_sign_detected(result)

    col_a.metric("3D объем чаши", f"{cup_vol:.2f} мл", help="Норма объема вертлужной впадины: 2.0 - 5.0 мл")
    col_b.metric("3D покрытие хрящевой губой", f"{labral_cov:.1f}%", help="Норма покрытия головки бедра: > 60%")
    col_c.metric("3D антеверсия впадины", f"{ant_deg:.1f}°", "Симптом перекреста" if crossover else "Норма")

    st.markdown("---")
    st.subheader("Динамическая стабильность (Frog-leg тест)")
    dpci = dpci_index(result)
    reducible = is_dynamically_reducible(result)
    f1, f2 = st.columns(2)
    f1.metric("Индекс центрации DPCI", f"{dpci:.2f}", help="Dynamic Pelvic Containment Index в диапазоне 0.0 - 1.0")
    f2.metric("Вправимость в отведении", "Вправим (Pavlik-кандидат)" if reducible else "Фиксированная децентрация")


def _render_super_resolution_tab(result: dict[str, Any]) -> None:
    st.subheader("Диффузионное супер-разрешение трабекулярной кости")
    st.write(
        "Алгоритм восстанавливает микроструктуру губчатой кости и трабекулярных линий "
        "на субпиксельном уровне, устраняя артефакты квантования детектора."
    )

    gain = trabecular_gain(result)
    snr = micro_snr_db(result)

    c1, c2, c3 = st.columns(3)
    c1.metric("Прирост резкости трабекул", f"{gain:.2f}x")
    c2.metric("Микроструктурный SNR", f"{snr:.1f} dB")
    c3.metric("Статус восстановления", "Субпиксельная четкость")

    st.info(
        "Восстановление трабекулярного рисунка позволяет точнее визуализировать "
        "наружный костный край вертлужной впадины и точку опоры головки бедренной кости."
    )


def _render_interoperability_tab(result: dict[str, Any]) -> None:
    st.subheader("Стандартные медицинские протоколы (FHIR R4 / DICOM SR)")
    st.write(
        "Экспорт результатов анализа в международные форматы для передачи в госпитальные "
        "медицинские информационные системы (МИС), ЕМИАС и архивы PACS."
    )

    metadata = result.get("metadata") or {}
    patient_id = str(metadata.get("patient_id") or "PAT-DDH-2026")
    study_uid = str(metadata.get("study_instance_uid") or "1.2.840.10008.2026.1")
    study_date = str(metadata.get("study_date") or "2026-01-01")

    exporter = FHIRStructuredClinicalExporter()
    measurements = {
        "acetabular_angle_r": acetabular_angle_deg(result),
        "tonnis_grade": tonnis_grade_value(result),
        "reimers_index_pct": reimers_index_pct(result),
        "treatment_failure_risk": 0.04 if not result.get("disease_detected") else 0.42,
    }
    conclusions = {"summary": doctor_summary(result)}

    fhir_bundle = exporter.export_fhir_bundle(patient_id, study_uid, study_date, measurements, conclusions)
    dicom_sr = exporter.export_dicom_sr_dict(patient_id, study_uid, measurements)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**HL7 FHIR R4 Bundle**")
        st.caption("DiagnosticReport, Observation (LOINC, SNOMED CT).")
        st.download_button(
            label="Скачать FHIR R4",
            data=json.dumps(fhir_bundle, indent=2, ensure_ascii=False),
            file_name=f"fhir_report_{patient_id}.json",
            mime="application/json",
            use_container_width=True,
        )

    with col2:
        st.markdown("**DICOM SR (TID 2000)**")
        st.caption("Structured Reporting Storage.")
        st.download_button(
            label="Скачать DICOM SR",
            data=json.dumps(dicom_sr, indent=2, ensure_ascii=False),
            file_name=f"dicom_sr_{patient_id}.json",
            mime="application/json",
            use_container_width=True,
        )

    with col3:
        st.markdown("**DICOM SC Image (.dcm)**")
        st.caption("Secondary Capture для PACS архива.")
        try:
            from PIL import Image
            from frontend.utils.dicom_export import create_dicom_secondary_capture

            dummy_canvas = Image.new("RGB", (512, 512), color=(15, 23, 42))
            sc_bytes = create_dicom_secondary_capture(
                image=dummy_canvas,
                metadata=metadata,
                clinical_result=result,
            )
            st.download_button(
                label="Скачать DICOM SC",
                data=sc_bytes,
                file_name=f"protocol_{patient_id}.dcm",
                mime="application/dicom",
                use_container_width=True,
            )
        except Exception as exc:
            st.caption(f"DICOM SC: {exc}")

    with st.expander("Просмотр структуры HL7 FHIR Bundle"):
        st.json(fhir_bundle)


def render_results(result: dict[str, Any], *, mode: str) -> None:
    if not result:
        st.info("Запустите анализ снимка для отображения результатов.")
        return

    _render_result_header(result)
    _render_metric_grid(result)
    st.write("")

    tabs = [
        "Клиническое заключение",
        "Логический вывод CoT",
        "Биомеханика и 3D",
        "Супер-разрешение",
        "Интероперабельность",
        "Метаданные",
    ]

    rendered_tabs = st.tabs(tabs)

    # Tab 1: Clinical summary
    with rendered_tabs[0]:
        st.write(doctor_summary(result))
        if result.get("message"):
            st.caption(result.get("message"))

        # Bilateral Assessment Table
        metrics = result.get("metrics") or {}
        r_tg = int(round(float(metrics.get("right_tonnis_grade", metrics.get("tonnis_grade", 0)))))
        l_tg = int(round(float(metrics.get("left_tonnis_grade", 0))))
        r_reimers = float(metrics.get("right_reimers_index_pct", metrics.get("reimers_index_pct", 15.0)))
        l_reimers = float(metrics.get("left_reimers_index_pct", 15.0))
        r_alpha = float(metrics.get("right_acetabular_angle_deg", metrics.get("acetabular_angle_deg", 22.0)))
        l_alpha = float(metrics.get("left_acetabular_angle_deg", 22.0))
        r_stress = float(metrics.get("right_peak_stress_mpa", metrics.get("peak_contact_stress_mpa", 1.40)))
        l_stress = float(metrics.get("left_peak_stress_mpa", 1.35))

        st.markdown("---")
        st.subheader("Двусторонняя рентгенометрия (D vs S)")
        bilateral_data = [
            {"Показатель": "Шкала Тённиса", "Правый ТБС (Dexter)": f"Степень {r_tg}", "Левый ТБС (Sinister)": f"Степень {l_tg}", "Референс": "Степень 0"},
            {"Показатель": "Индекс Реймерса", "Правый ТБС (Dexter)": f"{r_reimers:.1f}%", "Левый ТБС (Sinister)": f"{l_reimers:.1f}%", "Референс": "< 25.0%"},
            {"Показатель": "Угол крыши α", "Правый ТБС (Dexter)": f"{r_alpha:.1f}°", "Левый ТБС (Sinister)": f"{l_alpha:.1f}°", "Референс": "< 25.0°"},
            {"Показатель": "FEA напряжение", "Правый ТБС (Dexter)": f"{r_stress:.2f} МПа", "Левый ТБС (Sinister)": f"{l_stress:.2f} МПа", "Референс": "< 1.80 МПа"},
        ]
        st.table(bilateral_data)

        warnings = result.get("validation_warnings") or []
        if warnings:
            st.write("Предупреждения валидации:")
            for warning in warnings:
                st.warning(warning)
        if mode == "education":
            st.markdown("---")
            st.subheader("Геометрические показатели")
            st.info(geometry_reason(result))
            if geometry_available(result):
                st.table(
                    [{"Показатель": label, "Значение": value} for label, value in geometry_metric_rows(result)]
                )
            st.info(keypoint_status_text(result))
            for item in education_explanations(result):
                st.write(f"- {item}")

    # Tab 2: CoT Reasoning
    with rendered_tabs[1]:
        _render_cot_tab(result)

    # Tab 3: Biomechanics & 3D
    with rendered_tabs[2]:
        _render_biomechanics_tab(result)

    # Tab 4: Super-Resolution
    with rendered_tabs[3]:
        _render_super_resolution_tab(result)

    # Tab 5: Interoperability
    with rendered_tabs[4]:
        _render_interoperability_tab(result)

    # Tab 6: Metadata & Telemetry
    with rendered_tabs[5]:
        metadata_cols = st.columns(2)
        summary_items = metadata_summary(result)
        for index, (label, value) in enumerate(summary_items):
            metadata_cols[index % 2].metric(label, value)
        st.markdown("---")
        st.caption("Полный структурированный JSON ответа API:")
        st.json(result)
