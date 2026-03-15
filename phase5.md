# 🚀 Phase 5: Оптимизация, Submission и Финализация

## Цель Фазы
Оптимизировать систему, подготовить submission файлы (CSV + скриншоты), завершить документацию.

## Входные Данные
- Полностью рабочая система из Phase 4
- Тестовая выборка в data/test/

## Задачи AI

### 5.1 ONNX Conversion
```python
# scripts/convert_to_onnx.py
import torch
import onnx

def convert_to_onnx(model_path: str, output_path: str, input_shape: tuple):
    model = torch.load(model_path, map_location='cpu')
    model.eval()
    
    dummy_input = torch.randn(1, *input_shape)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        input_names=['input'],
        output_names=['output']
    )
```

### 5.2 CSV Submission Generator
```python
# scripts/generate_submission.py
import pandas as pd
from pathlib import Path

def generate_predictions_csv(test_dir: str, output_path: str):
    """
    Генерирует predictions.csv в формате ТЗ:
    id, class
    1OGQ64, 1
    28v1xk, 1
    ...
    """
    results = []
    
    for folder in Path(test_dir).iterdir():
        if folder.is_dir():
            # ID = имя папки
            object_id = folder.name
            
            # Найти DICOM файл в папке
            dicom_files = list(folder.glob('*.dcm'))
            if dicom_files:
                # Запустить инференс
                prediction = model.predict(dicom_files[0])
                class_label = 1 if prediction.disease_detected else 0
                
                results.append({
                    'id': object_id,
                    'class': class_label
                })
    
    # Создать DataFrame и сохранить
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False, header=False)
    
    print(f"Generated {len(results)} predictions")
    return output_path
```

### 5.3 Screenshot Generator
```python
# scripts/generate_screenshots.py
from pathlib import Path
import matplotlib.pyplot as plt

def generate_test_screenshots(test_dir: str, output_dir: str):
    """
    Генерирует скриншоты для каждого объекта тестовой выборки
    Формат: {id}.jpg (например: 1OGQ64.jpg)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for folder in Path(test_dir).iterdir():
        if folder.is_dir():
            object_id = folder.name
            dicom_files = list(folder.glob('*.dcm'))
            
            if dicom_files:
                # Запустить анализ
                result = analyze_image(dicom_files[0])
                
                # Создать визуализацию
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.imshow(result.image)
                
                # Добавить keypoints, lines, metrics
                visualize_result(ax, result)
                
                # Сохранить как {id}.jpg
                screenshot_path = Path(output_dir) / f"{object_id}.jpg"
                plt.savefig(screenshot_path, dpi=150, bbox_inches='tight')
                plt.close()
    
    print(f"Generated {len(list(Path(output_dir).glob('*.jpg')))} screenshots")
```

### 5.4 ID Format Verification
```python
# scripts/verify_id_format.py
def verify_submission_format(csv_path: str, screenshots_dir: str):
    """
    Проверяет что CSV и скриншоты соответствуют формату ТЗ
    """
    # Проверка CSV
    df = pd.read_csv(csv_path, header=None, names=['id', 'class'])
    
    # Проверка скриншотов
    screenshot_ids = [f.stem for f in Path(screenshots_dir).glob('*.jpg')]
    
    # Сравнение
    csv_ids = df['id'].tolist()
    
    missing_screenshots = set(csv_ids) - set(screenshot_ids)
    extra_screenshots = set(screenshot_ids) - set(csv_ids)
    
    return {
        'csv_count': len(csv_ids),
        'screenshot_count': len(screenshot_ids),
        'missing_screenshots': list(missing_screenshots),
        'extra_screenshots': list(extra_screenshots),
        'valid': len(missing_screenshots) == 0 and len(extra_screenshots) == 0
    }
```

### 5.5 Load Testing
```python
# tests/load_test.py
import time
import requests

def load_test(base_url: str, num_requests=100):
    latencies = []
    
    for i in range(num_requests):
        start = time.time()
        response = requests.post(f"{base_url}/api/v1/analyze", ...)
        end = time.time()
        latencies.append(end - start)
    
    return {
        'mean_latency': np.mean(latencies),
        'p95_latency': np.percentile(latencies, 95),
        'max_latency': np.max(latencies),
        'requests_per_second': num_requests / sum(latencies)
    }
```

### 5.6 Documentation Update
```markdown
# README.md — Обновить секции:
- Установка и запуск
- Структура проекта
- API документация
- Plugin guide
- Submission инструкции
```

### 5.7 Presentation Preparation
```markdown
# presentation/outline.md
1. Титульный слайд
2. Проблема
3. Решение
4. Архитектура
5. ML Pipeline
6. Метрики
7. UI Демонстрация
8. Explainable AI
9. Технические детали
10. Заключение + Q&A
```

### 5.8 Q&A Preparation
```markdown
# docs/qa_prep.md
20 вопросов и ответов из основного плана
```

## Критерии Завершения Фазы 5
- [ ] ONNX модели сконвертированы
- [ ] predictions.csv сгенерирован в формате ТЗ
- [ ] {id}.jpg скриншоты для всех объектов
- [ ] ID format verification passed
- [ ] Load test: API <1.5 сек средний
- [ ] README.md обновлен
- [ ] Презентация готова
- [ ] Q&A подготовка завершена
- [ ] GitHub release оформлен
- [ ] Файлы загружены в форму хакатона

## Выходные Артефакты
- `scripts/generate_submission.py`
- `scripts/generate_screenshots.py`
- `scripts/verify_id_format.py`
- `submissions/predictions.csv`
- `submissions/screenshots/{id}.jpg`
- `presentation/slides.pptx`
- `demo_video.mp4`
- `docs/qa_prep.md`

## Время на Выполнение
12 часов (День 6 + День 7)

## Финальный Чеклист
- [ ] Все тесты проходят
- [ ] Docker образ работает
- [ ] CSV загружен в форму
- [ ] Скриншоты загружены в форму
- [ ] Git репозиторий открыт
- [ ] Презентация готова
- [ ] 3 репетиции проведено

## Проект Завершён! 🎉
