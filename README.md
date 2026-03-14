# 🏥 ИТ+Мед 2026 — AI System for Medical Imaging

Система анализа медицинских снимков (DICOM) с plugin-архитектурой и ИИ-диагностикой.

## Стек

- **Backend:** Python 3.11, FastAPI, PyTorch, MONAI
- **Frontend:** Streamlit
- **Инфраструктура:** Docker, Docker Compose
- **Форматы данных:** DICOM (pydicom, SimpleITK)

## Архитектура

```
Frontend (Streamlit :8501)
    ↕ HTTP
API Gateway (FastAPI :8000)
    ↕
Core (Plugin Manager)
    ↕
Plugins (ML Models)
```

## Быстрый старт

```bash
# Клонировать репозиторий
git clone https://github.com/ArtemChik103/ITmed.git
cd ITmed

# Запустить все сервисы
docker-compose up --build
```

После запуска:
- **API:** http://localhost:8000/health
- **UI:** http://localhost:8501
- **API Docs:** http://localhost:8000/docs

## Структура проекта

```
it-med-2026/
├── api/           # FastAPI backend
├── core/          # DICOM loader, Plugin Manager
├── frontend/      # Streamlit UI
├── tests/         # Pytest тесты
├── data/          # DICOM данные (не коммитятся)
├── plugins/       # ML-плагины (Phase 2+)
├── models/        # Архитектуры моделей (Phase 2+)
└── weights/       # Веса моделей (Phase 2+)
```

## Датасет

- `train/Норма/` — нормальные снимки
- `train/Патология/` — патологические снимки
- `test_done/` — тестовые снимки

## Этапы разработки

- [x] **Phase 1:** Инфраструктура и фундамент
- [ ] **Phase 2:** ML-модели и plugin-архитектура
- [ ] **Phase 3:** Продвинутая диагностика
- [ ] **Phase 4:** Production и мониторинг
