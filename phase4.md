# 🎨 Phase 4: Frontend — UI для Врача и Обучения

## Цель Фазы
Создать пользовательский интерфейс в Streamlit с двумя режимами: врач и обучение.

## Входные Данные
- ML модели из Phase 3
- API endpoints из Phase 2

## Задачи AI

### 4.1 Streamlit Main App
```python
# frontend/app.py
import streamlit as st

def main():
    st.set_page_config(page_title="ИТ+Мед 2026", layout="wide")
    
    mode = st.sidebar.selectbox("Режим", ["Врач", "Обучение"])
    
    if mode == "Врач":
        doctor_mode()
    else:
        education_mode()

if __name__ == "__main__":
    main()
```

### 4.2 File Upload Component
```python
# frontend/components/upload.py
def file_uploader():
    uploaded_file = st.file_uploader(
        "Загрузите DICOM файл",
        type=["dcm", "dicom"],
        help="Поддерживаются только рентгеновские снимки"
    )
    return uploaded_file
```

### 4.3 API Client
```python
# frontend/api_client.py
import requests

class APIClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def analyze(self, file, plugin_type, mode):
        response = requests.post(
            f"{self.base_url}/api/v1/analyze",
            files={"file": file},
            data={"plugin_type": plugin_type, "mode": mode}
        )
        return response.json()
```

### 4.4 Result Display Component
```python
# frontend/components/results.py
def display_results(analysis_result: dict):
    st.metric("Уверенность", f"{analysis_result['confidence']:.2%}")
    st.metric("Диагноз", "ПАТОЛОГИЯ" if analysis_result['disease_detected'] else "НОРМА")
    
    st.subheader("Метрики")
    for metric, value in analysis_result['metrics'].items():
        st.write(f"{metric}: {value}")
```

### 4.5 Image Viewer с Visualizations
```python
# frontend/components/viewer.py
import plotly.graph_objects as go

def display_image_with_overlay(image, keypoints, lines):
    fig = go.Figure()
    fig.add_trace(go.Image(z=image))
    # Добавить keypoints и lines
    st.plotly_chart(fig)
```

### 4.6 Grad-CAM Heatmap Component
```python
# frontend/components/heatmap.py
def display_heatmap(original_image, heatmap, alpha=0.5):
    overlay = cv2.addWeighted(original_image, 1-alpha, heatmap, alpha, 0)
    st.image(overlay, caption="Grad-CAM Heatmap")
```

### 4.7 PDF Export
```python
# frontend/utils/pdf_export.py
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table

def generate_pdf(analysis_result, patient_info, output_path):
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    # Генерация отчета
    doc.build(elements)
```

### 4.8 Mode Switching Logic
```python
# frontend/app.py
def switch_mode(current_mode, new_mode):
    st.session_state.mode = new_mode
    st.rerun()
```

### 4.9 Educational Tooltips
```python
# frontend/data/tooltips.json
{
    "alpha_angle": {
        "description": "Ацетабулярный угол",
        "normal_range": "25-35 градусов",
        "pathology_threshold": ">35 градусов"
    }
}
```

### 4.10 Session Management
```python
# frontend/session_manager.py
def save_to_session(analysis_result):
    if 'history' not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append(analysis_result)
```

## Критерии Завершения Фазы 4
- [ ] Streamlit app запускается
- [ ] Загрузка файлов работает
- [ ] API integration работает
- [ ] Результат отображается корректно
- [ ] Heatmap генерируется <2 сек
- [ ] PDF экспорт работает
- [ ] Переключение режимов <1 сек
- [ ] Session history сохраняется

## Выходные Артефакты
- `frontend/app.py`
- `frontend/components/` — все компоненты
- `frontend/utils/pdf_export.py`
- `frontend/data/tooltips.json`
- `frontend/api_client.py`

## Время на Выполнение
16 часов (День 4 + День 5)

## Следующая Фаза
Перейти к phase5.md после успешного завершения всех чеклистов.
