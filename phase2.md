# 🏗️ Phase 2: Ядро Системы — Plugin Architecture и DICOM Processing

## Цель Фазы
Реализовать базовое ядро системы: IPlugin interface, Plugin Manager, DICOM loader с поддержкой X-ray.

## Входные Данные
- Структура из Phase 1
- Распакованные DICOM файлы в data/training/ и data/test/

## Задачи AI

### 2.1 Создание IPlugin Interface
```python
# core/plugin_manager.py
from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Dict, List, Tuple, Optional
import numpy as np

class AnalysisResult(BaseModel):
    disease_detected: bool
    confidence: float
    metrics: Dict[str, float]
    keypoints: List[Tuple[float, float]]
    heatmap_url: Optional[str]
    processing_time_ms: int

class PluginMetadata(BaseModel):
    name: str
    version: str
    description: str
    supported_modalities: List[str]

class IPlugin(ABC):
    @abstractmethod
    async def load_model(self) -> None:
        """Загрузка модели в память"""
        pass
    
    @abstractmethod
    async def preprocess(self, image: np.ndarray, metadata: dict) -> np.ndarray:
        """Препроцессинг изображения"""
        pass
    
    @abstractmethod
    async def analyze(self, image: np.ndarray) -> AnalysisResult:
        """Анализ изображения и возврат результатов"""
        pass
    
    @abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """Метаданные плагина"""
        pass
```

### 2.2 Создание Plugin Manager
```python
# core/plugin_registry.py
class PluginRegistry:
    def __init__(self):
        self.plugins: Dict[str, IPlugin] = {}
    
    def register(self, name: str, plugin: IPlugin):
        self.plugins[name] = plugin
    
    def get(self, name: str) -> IPlugin:
        return self.plugins.get(name)
    
    def list_plugins(self) -> List[str]:
        return list(self.plugins.keys())
```

### 2.3 Создание DICOM Loader (X-ray Specific)
```python
# core/dicom_loader.py
import pydicom
import numpy as np
from typing import Tuple, Dict
from pathlib import Path

def load_dicom(file_path: str) -> Tuple[np.ndarray, dict]:
    """
    Загружает DICOM файл (рентген) с обязательной экстракцией Pixel Spacing
    """
    ds = pydicom.dcmread(file_path)
    
    # Проверка Pixel Spacing
    if not hasattr(ds, 'PixelSpacing'):
        pixel_spacing = [1.0, 1.0]  # Default
    else:
        pixel_spacing = [float(x) for x in ds.PixelSpacing]
    
    # Извлечение изображения
    image = ds.pixel_array.astype(np.float32)
    
    # Нормализация для рентгена
    if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
        image = image * ds.RescaleSlope + ds.RescaleIntercept
    
    metadata = {
        'pixel_spacing_mm': pixel_spacing,
        'patient_id': getattr(ds, 'PatientID', 'Unknown'),
        'study_date': getattr(ds, 'StudyDate', 'Unknown'),
        'modality': getattr(ds, 'Modality', 'XR'),
        'image_shape': image.shape,
        'bits_allocated': getattr(ds, 'BitsAllocated', 16)
    }
    
    return image, metadata
```

### 2.4 Создание DICOM Validator
```python
# core/dicom_validator.py
class DICOMValidator:
    required_tags = ['PatientID', 'StudyDate', 'Modality', 'PixelSpacing']
    
    def validate(self, file_path: str) -> dict:
        ds = pydicom.dcmread(file_path)
        missing = [tag for tag in self.required_tags if not hasattr(ds, tag)]
        return {
            'valid': len(missing) == 0,
            'missing_tags': missing,
            'warnings': self._check_warnings(ds)
        }
```

### 2.5 Создание MONAI Preprocessor
```python
# core/preprocessor.py
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged, Spacingd, Orientationd

def get_preprocessing_pipeline():
    return Compose([
        EnsureChannelFirstd(keys=["image"]),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0)),
        Orientationd(keys=["image"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0),
    ])
```

### 2.6 Создание Data Quality Check Script
```python
# scripts/data_quality_check.py
# (Полный код из основного плана)
```

### 2.7 Создание ID Format Verification Script
```python
# scripts/verify_id_format.py
import os
from pathlib import Path

def verify_test_ids(test_dir: str, expected_format: str = "folder_name"):
    """
    Проверяет что ID объектов соответствуют формату ТЗ (имена папок)
    """
    ids = []
    for item in Path(test_dir).iterdir():
        if item.is_dir():
            ids.append(item.name)
    
    # Проверка формата (например: 1OGQ64, 28v1xk, и т.д.)
    valid_count = sum(1 for id in ids if len(id) == 6 and id.isalnum())
    
    return {
        'total_ids': len(ids),
        'valid_format': valid_count == len(ids),
        'ids': ids,
        'sample': ids[:5]
    }
```

## Критерии Завершения Фазы 2
- [ ] IPlugin interface создан и типизирован
- [ ] Plugin Registry работает
- [ ] DICOM loader извлекает Pixel Spacing
- [ ] Validator проверяет обязательные теги
- [ ] Preprocessing pipeline настроен
- [ ] Data quality check запускается
- [ ] ID format verification работает

## Выходные Артефакты
- `core/plugin_manager.py`
- `core/plugin_registry.py`
- `core/dicom_loader.py`
- `core/dicom_validator.py`
- `core/preprocessor.py`
- `scripts/data_quality_check.py`
- `scripts/verify_id_format.py`

## Время на Выполнение
8 часов (День 0, 14:00-20:00 + День 1, 09:00-13:00)

## Следующая Фаза
Перейти к phase3.md после успешного завершения всех чеклистов.
