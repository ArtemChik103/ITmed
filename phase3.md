# 🧠 Phase 3: ML Модуль — Классификация и Detection

## Цель Фазы
Обучить и интегрировать модели машинного обучения: классификатор патологии и детектор ключевых точек.

## Входные Данные
- Ядро системы из Phase 2
- Подготовленные данные в data/training/

## Задачи AI

### 3.1 Dataset Split Script
```python
# data/split_dataset.py
from sklearn.model_selection import train_test_split
import shutil
from pathlib import Path

def split_dataset(data_dir: str, ratios=(0.7, 0.15, 0.15)):
    """
    Разделение датасета 70/15/15 со стратификацией
    """
    # Реализация разделения с сохранением структуры папок
    pass
```

### 3.2 Augmentation Pipeline
```python
# data/augmentations.py
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_augments():
    return A.Compose([
        A.Rotate(limit=15, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.ShiftScaleRotate(p=0.3),
        ToTensorV2()
    ])
```

### 3.3 Classifier Model (ResNet50 + FocalLoss)
```python
# models/classifier.py
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class HipDysplasiaClassifier(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        self.backbone.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )
    
    def forward(self, x):
        return self.backbone(x)
```

### 3.4 FocalLoss Implementation
```python
# models/losses.py
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        BCE_loss = nn.BCEWithLogitsLoss(reduce=False)(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return torch.mean(F_loss)
```

### 3.5 Training Loop с 5-Fold CV
```python
# train/classifier_train.py
from sklearn.model_selection import StratifiedKFold

def train_with_cv(data_dir: str, n_folds=5):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        # Обучение модели
        # Валидация
        # Сохранение checkpoint
        pass
```

### 3.6 Keypoint Detector (HRNet/U-Net)
```python
# models/keypoint_detector.py
class KeypointDetector(nn.Module):
    def __init__(self, num_keypoints=4):
        super().__init__()
        # HRNet или U-Net архитектура
        self.num_keypoints = num_keypoints
    
    def forward(self, x):
        # Возвращает heatmap для каждой ключевой точки
        pass
```

### 3.7 Metrics Calculation (Хингельрейнер)
```python
# plugins/hip_dysplasia/metrics.py
def calculate_hilgenreiner_metrics(keypoints: dict, pixel_spacing: List[float]) -> dict:
    """
    Расчет углов и расстояний по Хингельрейнеру в миллиметрах
    """
    # Реализация геометрии
    pass
```

### 3.8 Evaluation Script
```python
# train/evaluate.py
from sklearn.metrics import sensitivity_score, specificity_score, f1_score, roc_auc_score

def evaluate_model(model, test_loader):
    metrics = {
        'sensitivity': sensitivity_score(y_true, y_pred),
        'specificity': specificity_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_scores)
    }
    return metrics
```

## Критерии Завершения Фазы 3
- [ ] Dataset split выполнен (70/15/15)
- [ ] Augmentation pipeline работает
- [ ] Classifier обучен с 5-Fold CV
- [ ] Sensitivity ≥95% на validation
- [ ] Keypoint detector обучен
- [ ] Метрики Хингельрейнера рассчитываются в мм
- [ ] Evaluation script генерирует отчет

## Выходные Артефакты
- `data/split_dataset.py`
- `data/augmentations.py`
- `models/classifier.py`
- `models/losses.py`
- `models/keypoint_detector.py`
- `train/classifier_train.py`
- `train/evaluate.py`
- `plugins/hip_dysplasia/metrics.py`
- `models/checkpoints/` — сохраненные веса

## Время на Выполнение
16 часов (День 2 + День 3)

## Следующая Фаза
Перейти к phase4.md после успешного завершения всех чеклистов.
