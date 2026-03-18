# ИТ+Мед 2026

Репозиторий для финальной сдачи системы анализа DICOM-снимков тазобедренных суставов. Текущая версия построена по принципу `classifier-first`: итоговый бинарный verdict дает classifier runtime, а keypoints используются только как explainability layer в режиме обучения.

## Что делает система

- принимает DICOM-снимок таза;
- валидирует метаданные и запускает plugin `hip_dysplasia`;
- выдает бинарный класс `0/1`, confidence, threshold и служебные флаги runtime;
- в режиме `education` может показывать keypoint overlay и расширенный PDF;
- для набора `test_done` формирует итоговый пакет результатов в текстовом и машинно-проверяемом виде.

## Режимы работы

- `doctor`: короткая сводка для врача, без перегрузки интерфейса.
- `education`: тот же classifier verdict плюс anatomy overlay и подробный JSON/PDF-слой.

Важно:

- keypoints не меняют `class`, `disease_detected`, `confidence` и `threshold`;
- quantitative geometry автоматически не рассчитывается;
- причина в том, что семантика raw MTDDH keypoints пока не валидирована для клинических вычислений;
- fallback-режим сохраняет работоспособность pipeline, но честно помечается как `non-diagnostic`.

## Что смотреть эксперту

Основные финальные артефакты:

- репозиторий: `https://github.com/ArtemChik103/ITmed`
- файл классов: [deliverables/predictions.csv](/C:/Users/pvppv/Desktop/roo/it-med-2026/deliverables/predictions.csv)
- архив результатов: [deliverables/results_test_done.zip](/C:/Users/pvppv/Desktop/roo/it-med-2026/deliverables/results_test_done.zip)
- PDF со слайдами: [deliverables/presentation.pdf](/C:/Users/pvppv/Desktop/roo/it-med-2026/deliverables/presentation.pdf)
- реестр состава repo: [docs/final_repo_registry.md](/C:/Users/pvppv/Desktop/roo/it-med-2026/docs/final_repo_registry.md)

Если нужно быстро проверить только содержимое результатов:

- откройте `deliverables/results_test_done/summary.csv`;
- затем при необходимости конкретный `reports/{id}.json` или `reports/{id}.txt`;
- `predictions.csv` остается отдельным обязательным deliverable в формате `id,class`.

## Структура проекта

- [api/main.py](/C:/Users/pvppv/Desktop/roo/it-med-2026/api/main.py) и [api/schemas.py](/C:/Users/pvppv/Desktop/roo/it-med-2026/api/schemas.py): FastAPI backend и typed schema.
- [core/](/C:/Users/pvppv/Desktop/roo/it-med-2026/core): загрузка DICOM, валидация и preprocessing.
- [plugins/hip_dysplasia/plugin.py](/C:/Users/pvppv/Desktop/roo/it-med-2026/plugins/hip_dysplasia/plugin.py): classifier-first plugin runtime.
- [plugins/hip_dysplasia/keypoint_runtime.py](/C:/Users/pvppv/Desktop/roo/it-med-2026/plugins/hip_dysplasia/keypoint_runtime.py): optional keypoint runtime.
- [frontend/app.py](/C:/Users/pvppv/Desktop/roo/it-med-2026/frontend/app.py): Streamlit frontend.
- [frontend/utils/pdf_export.py](/C:/Users/pvppv/Desktop/roo/it-med-2026/frontend/utils/pdf_export.py): генерация PDF-отчета.
- [scripts/export_test_done_reports.py](/C:/Users/pvppv/Desktop/roo/it-med-2026/scripts/export_test_done_reports.py): единый batch pipeline по `test_done`.
- [scripts/generate_presentation_pdf.py](/C:/Users/pvppv/Desktop/roo/it-med-2026/scripts/generate_presentation_pdf.py): генерация `presentation.pdf`.

## Быстрый запуск

### Установка

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Локальный API

```bash
set HIP_DYSPLASIA_MODEL_MANIFEST=models/checkpoints/resnet50_bce_v1/model_manifest.json
set HIP_DYSPLASIA_KEYPOINT_CHECKPOINT=models/checkpoints/resnet50_mtddh_keypoints_v1/best.ckpt
set HIP_DYSPLASIA_KEYPOINT_DEVICE=auto
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Frontend

```bash
set API_URL=http://127.0.0.1:8000
streamlit run frontend/app.py
```

### Docker Compose

```bash
docker compose up -d --build api frontend
```

## Финальная выгрузка по `test_done`

Один сценарий собирает все обязательные результаты:

```bash
python scripts/export_test_done_reports.py ^
  --test-root ../test_done ^
  --output-dir deliverables/results_test_done ^
  --predictions-output deliverables/predictions.csv ^
  --zip-output deliverables/results_test_done.zip ^
  --manifest-path models/checkpoints/resnet50_bce_v1/model_manifest.json ^
  --keypoint-checkpoint models/checkpoints/resnet50_mtddh_keypoints_v1/best.ckpt
```

Что создается:

- `deliverables/predictions.csv`
- `deliverables/results_test_done/summary.csv`
- `deliverables/results_test_done/reports/{id}.json`
- `deliverables/results_test_done/reports/{id}.txt`
- `deliverables/results_test_done/README_results.txt`
- `deliverables/results_test_done.zip`

Проверка `id,class`:

```bash
python scripts/verify_id_format.py ^
  --test-root ../test_done ^
  --csv deliverables/predictions.csv ^
  --check-sorted
```

## PDF со слайдами

```bash
python scripts/generate_presentation_pdf.py ^
  --test-root ../test_done ^
  --output deliverables/presentation.pdf
```

PDF сделан коротким под защиту на 5 минут: задача, архитектура, classifier runtime, explainability, ограничение по geometry, pipeline по `test_done` и итоговые deliverables.

## Тесты

```bash
pytest -q
pytest tests/test_scripts.py -q
```

## Ограничения

- принимаются только `.dcm` и `.dicom`;
- classifier runtime остается главным источником verdict;
- keypoints не расширяют clinical claim;
- quantitative geometry автоматически не публикуется, если она не валидирована;
- если веса недоступны, система остается рабочей, но результат нужно трактовать как технический fallback.
