# 🏗️ Phase 1: Инициализация Проекта и Базовая Инфраструктура

## Цель Фазы
Создать фундамент проекта: репозиторий, структуру папок, базовые конфигурации, Docker-окружение.

## Входные Данные
- Доступ к Telegram-каналу хакатона (пароли на архивы)
- Ссылки на датасеты (Yandex Disk)
- Python 3.11.8 установлен

## Задачи AI

### 1.1 Создание Структуры Репозитория
```bash
# Создать корневую структуру
mkdir -p it-med-2026/{api,core,models,plugins,frontend,data,train,scripts,tests,docs,presentation,assets,submissions}
mkdir -p it-med-2026/plugins/{hip_dysplasia,lung_pneumonia,template}
mkdir -p it-med-2026/frontend/{components,utils,data}
mkdir -p it-med-2026/.github/workflows
mkdir -p it-med-2026/submissions/screenshots
```

### 1.2 Инициализация Git
```bash
cd it-med-2026
git init
git branch -M main
```

### 1.3 Создание .gitignore
```
# AI должен создать .gitignore со следующим содержимым:
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/
*.egg-info/
dist/
build/
*.egg
*.log
.DS_Store
*.ipynb_checkpoints
.idea/
.vscode/
*.swp
*.swo
*~
data/raw/
data/processed/
models/checkpoints/
*.pth
*.onnx
.env
*.pem
*.key
submissions/
demo_video.mp4
```

### 1.4 Создание requirements.txt
```
# AI должен создать requirements.txt с точными версиями из раздела "Стек Технологий" выше
python==3.11.8
pytorch==2.2.0+cu121
...
```

### 1.5 Создание Dockerfile
```dockerfile
# Multi-stage build с CUDA support
FROM python:3.11.8-slim as base
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000 8501
CMD ["bash", "start.sh"]
```

### 1.6 Создание docker-compose.yml
```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
  frontend:
    build: .
    ports:
      - "8501:8501"
    depends_on:
      - api
```

### 1.7 Распаковка Данных
```bash
# scripts/extract_data.sh
#!/bin/bash
echo "Скачивание архивов..."
# Скачать с Yandex Disk (нужен cookie или прямой линк)
echo "Распаковка с паролями из Telegram..."
unzip -P $TRAINING_PASSWORD training_data.zip -d data/training/
unzip -P $TEST_PASSWORD test_data.zip -d data/test/
echo "Готово!"
```

### 1.8 Создание README.md
```markdown
# ИТ+Мед 2026 — ИИ-Система Диагностики

## Быстрый старт
```bash
docker-compose up --build
```

## Структура
[Описание из раздела выше]

## Требования
Python 3.11.8, Docker, CUDA 12.1
```

## Критерии Завершения Фазы 1
- [ ] Репозиторий создан и инициализирован
- [ ] Все директории существуют
- [ ] .gitignore корректный
- [ ] requirements.txt создан
- [ ] Dockerfile и docker-compose.yml рабочие
- [ ] Данные распакованы и доступны в data/
- [ ] README.md содержит инструкцию по запуску

## Выходные Артефакты
- `it-med-2026/` — корневая директория проекта
- `data/training/` — обучающая выборка
- `data/test/` — тестовая выборка
- `.git/` — инициализированный репозиторий

## Время на Выполнение
4 часа (День 0, 09:00-13:00)

## Следующая Фаза
Перейти к phase2.md после успешного завершения всех чеклистов.
