# 🏥 ИТ+Мед 2026 — AI System for Medical Imaging

Система анализа медицинских снимков (DICOM) с plugin-архитектурой, базовым X-ray pipeline и двумя режимами UI.

## Стек

- **Backend:** Python 3.11, FastAPI, PyTorch, MONAI
- **Frontend:** Streamlit
- **Core:** plugin registry, DICOM loader, validator, preprocessor
- **Инфраструктура:** Docker, Docker Compose, GitHub Actions
- **Форматы данных:** DICOM (pydicom, SimpleITK)

## Архитектура

```text
Frontend (Streamlit :8501)
    ↕ HTTP
API Gateway (FastAPI :8000)
    ↕
Core (Plugin Manager + Validator + Preprocessor)
    ↕
Plugins (Phase 2 baseline: hip_dysplasia)
```

## Данные

Проект использует внешние датасеты из соседних директорий workspace:

- `../train` — обучающая выборка
- `../test_done` — тестовая выборка

В Docker эти директории монтируются только на чтение:

- `/datasets/train`
- `/datasets/test`

Это каноническая схема для текущего репозитория: данные не копируются внутрь Git.

## Быстрый старт

```bash
cd it-med-2026
docker compose up --build
```

После запуска:

- **API health:** `http://localhost:8000/health`
- **API docs:** `http://localhost:8000/docs`
- **Plugins:** `http://localhost:8000/api/v1/plugins`
- **UI:** `http://localhost:8501`

## API

### `GET /health`

Возвращает:

```json
{"status":"ok","version":"1.0.0"}
```

### `GET /api/v1/plugins`

Возвращает список зарегистрированных плагинов.

### `POST /api/v1/analyze`

Параметры query:

- `plugin_type=hip_dysplasia`
- `mode=doctor|education`

Файл:

- `file`: `.dcm` или `.dicom`

Phase 2 сейчас подключает baseline-plugin `hip_dysplasia`, который честно помечает ответ как pre-ML/heuristic и используется для проверки pipeline.

Поддерживаемые проекционные модальности baseline pipeline: `DX`, `CR`, `XR`, `RG`, `RF`. Для `RF` текущий контракт ограничен single-frame grayscale DICOM.

## Проверка Phase 1 / Phase 2

### 1. Поднять контейнеры

```bash
docker compose up --build -d
```

### 2. Проверить health

```bash
python - <<'PY'
import urllib.request
print(urllib.request.urlopen("http://127.0.0.1:8000/health").read().decode())
PY
```

### 3. Проверить frontend

Откройте `http://localhost:8501`.

### 4. Прогнать тесты в контейнере

```bash
docker compose exec -T api pytest -q
```

### 5. Проверить DICOM loader вручную

```bash
python -m core.dicom_loader path/to/file.dcm
```

### 6. Проверить mixed-layout ID из тестовой выборки

```bash
python scripts/verify_id_format.py --test-root ../test_done
```

### 7. Собрать data quality report

```bash
python scripts/data_quality_check.py --train-root ../train --test-root ../test_done --output docs/data_quality_report.md
```

## Структура проекта

```text
it-med-2026/
├── .github/workflows/         # CI
├── api/                       # FastAPI backend
├── core/                      # DICOM loader, validator, preprocessor, plugin manager
├── docs/                      # Документация и отчеты
├── frontend/                  # Streamlit UI
├── models/                    # ML-модели и будущие архитектуры
├── plugins/                   # Плагины анализа
├── scripts/                   # CLI-утилиты и служебные скрипты
├── submissions/               # Submission-артефакты
├── tests/                     # Pytest тесты
├── weights/                   # Веса моделей
├── Dockerfile
├── docker-compose.yml
├── start.sh
├── phase1.md
├── phase2.md
├── phase3.md
├── phase4.md
└── phase5.md
```

## Этапы разработки

- [x] **Phase 1:** Инфраструктура и фундамент
- [x] **Phase 2:** Ядро системы, плагины и DICOM processing
- [ ] **Phase 3:** ML pipeline, классификация и ключевые точки
- [ ] **Phase 4:** UI для врача и режима обучения
- [ ] **Phase 5:** Оптимизация, submission и финализация

## Phase-документы

- [phase1.md](phase1.md) — полное описание Phase 1
- [phase2.md](phase2.md) — полное описание Phase 2
- [phase3.md](phase3.md) — полное описание Phase 3
- [phase4.md](phase4.md) — полное описание Phase 4
- [phase5.md](phase5.md) — полное описание Phase 5
