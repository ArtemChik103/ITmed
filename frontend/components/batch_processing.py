"""Component for batch processing multiple DICOM and image files with progress tracking and CSV export."""
from __future__ import annotations

import io
import time
import zipfile
from typing import Any

import pandas as pd
import streamlit as st

from frontend.utils.api_client import ApiClient, ApiClientError


def extract_files_from_zip(zip_bytes: bytes) -> list[tuple[str, bytes]]:
    """Extract valid radiological image files from a ZIP archive."""
    extracted: list[tuple[str, bytes]] = []
    valid_extensions = {".dcm", ".dicom", ".png", ".jpg", ".jpeg"}
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for member in zf.namelist():
            if member.startswith("__MACOSX/") or member.endswith("/"):
                continue
            lower_name = member.lower()
            if any(lower_name.endswith(ext) for ext in valid_extensions):
                filename = member.split("/")[-1]
                if filename:
                    extracted.append((filename, zf.read(member)))
    return extracted


def render_batch_processing(api_url: str, plugin_type: str = "hip_dysplasia") -> None:
    """Render the batch processing tab interface."""
    st.markdown(
        """
        <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 12px; padding: 1.2rem; margin-bottom: 1.5rem;">
            <h3 style="margin-top:0; color: #38bdf8;">Пакетная диагностика рентгенограмм (Batch Mode)</h3>
            <p style="color: #94a3b8; font-size: 0.95rem; margin-bottom: 0;">
                Массовая обработка рентгенологических исследований для скрининга отделений и ретроспективного анализа.
                Поддерживается прямая загрузка нескольких файлов DICOM (.dcm), изображений (.png/.jpg) или ZIP-архивов с исследованиями.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Выберите несколько файлов DICOM / изображений или ZIP-архив с исследованиями",
        type=["dcm", "dicom", "zip", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        help="Загрузите один или несколько файлов DICOM, либо ZIP архив с серией рентгенограмм",
        key="batch_file_uploader",
    )

    if not uploaded_files:
        st.info("Загрузите файлы для запуска массовой автоматической обработки.")
        return

    # Expand any zip files and collect all file items: (filename, bytes)
    items_to_process: list[tuple[str, bytes]] = []
    for f in uploaded_files:
        raw_bytes = f.getvalue()
        if f.name.lower().endswith(".zip"):
            extracted = extract_files_from_zip(raw_bytes)
            items_to_process.extend(extracted)
        else:
            items_to_process.append((f.name, raw_bytes))

    st.markdown(
        f"**Всего готово к обработке:** `{len(items_to_process)}` файлов "
        f"(из {len(uploaded_files)} загруженных объектов)."
    )

    client = ApiClient(api_url)

    c_run, _ = st.columns([1, 2])
    with c_run:
        start_batch = st.button("Запустить пакетную диагностику", type="primary", use_container_width=True)

    if "batch_results" not in st.session_state:
        st.session_state.batch_results = []

    if start_batch:
        st.session_state.batch_results = []
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        results_list: list[dict[str, Any]] = []
        total = len(items_to_process)
        start_time_all = time.perf_counter()

        for idx, (filename, file_bytes) in enumerate(items_to_process):
            status_text.text(f"Обработка [{idx + 1}/{total}]: {filename}...")
            t0 = time.perf_counter()
            try:
                res = client.analyze(
                    file_bytes=file_bytes,
                    filename=filename,
                    plugin_type=plugin_type,
                    mode="education",
                )
                duration_ms = (time.perf_counter() - t0) * 1000.0

                meta = res.get("metadata") or {}
                metrics = res.get("metrics") or {}
                patient_id = meta.get("patient_id") or f"PAT-{idx + 1:04d}"
                disease_detected = bool(res.get("disease_detected", False))
                disease_prob = float(res.get("confidence", metrics.get("model_probability", 0.0)))

                # Bilateral metrics from plugin metrics dictionary
                r_tonnis = int(round(float(metrics.get("right_tonnis_grade", metrics.get("tonnis_grade", 0)))))
                l_tonnis = int(round(float(metrics.get("left_tonnis_grade", 0))))

                r_angle = float(metrics.get("right_acetabular_angle_deg", metrics.get("acetabular_angle_deg", 0.0)))
                l_angle = float(metrics.get("left_acetabular_angle_deg", 0.0))

                r_reimers = float(metrics.get("right_reimers_index_pct", metrics.get("reimers_index_pct", 0.0)))
                l_reimers = float(metrics.get("left_reimers_index_pct", 0.0))

                r_stress = float(metrics.get("right_peak_stress_mpa", metrics.get("peak_contact_stress_mpa", 0.0)))
                l_stress = float(metrics.get("left_peak_stress_mpa", 0.0))

                results_list.append(
                    {
                        "Имя файла": filename,
                        "ID пациента": patient_id,
                        "Диагноз": "Патология (ДТБС)" if disease_detected else "Норма",
                        "Вероятность (%)": round(disease_prob * 100.0, 1),
                        "Тённис (D)": f"Степень {r_tonnis}",
                        "Тённис (S)": f"Степень {l_tonnis}",
                        "Угол крыши D (°)": round(r_angle, 1),
                        "Угол крыши S (°)": round(l_angle, 1),
                        "Реймерс D (%)": round(r_reimers, 1),
                        "Реймерс S (%)": round(l_reimers, 1),
                        "FEA Напряжение D (МПа)": round(r_stress, 2),
                        "FEA Напряжение S (МПа)": round(l_stress, 2),
                        "Время (мс)": round(duration_ms, 1),
                        "raw_result": res,
                    }
                )
            except ApiClientError as err:
                results_list.append(
                    {
                        "Имя файла": filename,
                        "ID пациента": "Ошибка",
                        "Диагноз": f"Ошибка: {err}",
                        "Вероятность (%)": 0.0,
                        "Тённис (D)": "-",
                        "Тённис (S)": "-",
                        "Угол крыши D (°)": 0.0,
                        "Угол крыши S (°)": 0.0,
                        "Реймерс D (%)": 0.0,
                        "Реймерс S (%)": 0.0,
                        "FEA Напряжение D (МПа)": 0.0,
                        "FEA Напряжение S (МПа)": 0.0,
                        "Время (мс)": round((time.perf_counter() - t0) * 1000.0, 1),
                        "raw_result": None,
                    }
                )

            progress_bar.progress((idx + 1) / float(total))

        total_duration = time.perf_counter() - start_time_all
        status_text.success(f"Пакетная диагностика завершена за {total_duration:.2f} с ({len(results_list)} исследований).")
        st.session_state.batch_results = results_list

    # Render summary table and metrics if results are available
    if st.session_state.batch_results:
        batch_data = st.session_state.batch_results
        df = pd.DataFrame(
            [
                {k: v for k, v in item.items() if k != "raw_result"}
                for item in batch_data
            ]
        )

        total_cases = len(df)
        pathology_cases = sum(1 for item in batch_data if "Патология" in str(item.get("Диагноз")))
        normal_cases = sum(1 for item in batch_data if "Норма" in str(item.get("Диагноз")))
        avg_time = df["Время (мс)"].mean() if not df.empty else 0.0

        st.markdown("### Сводная статистика скрининга")
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Всего исследований", f"{total_cases}")
        kpi2.metric("Выявлено патологий", f"{pathology_cases}", delta=f"{pathology_cases / max(total_cases, 1) * 100:.1f}%")
        kpi3.metric("Норма (без патологии)", f"{normal_cases}", delta=f"{normal_cases / max(total_cases, 1) * 100:.1f}%")
        kpi4.metric("Ср. время на снимок", f"{avg_time:.1f} мс")

        st.markdown("### Таблица результатов")
        st.dataframe(df, use_container_width=True, hide_index=True)

        csv_data = df.to_csv(index=False, encoding="utf-8-sig")
        c_csv, _ = st.columns([1, 2])
        with c_csv:
            st.download_button(
                label="Скачать реестр результатов (CSV)",
                data=csv_data,
                file_name=f"batch_screening_results_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True,
            )
