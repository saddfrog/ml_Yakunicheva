"""Конфигурация эксперимента для OGYEIv2"""

from dataclasses import dataclass
from pathlib import Path

# Пути
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ogyieiv2"
ARTIFACTS_PATH = PROJECT_ROOT / "artifacts"

# Создаём папки для артефактов
(ARTIFACTS_PATH / "metrics").mkdir(parents=True, exist_ok=True)
(ARTIFACTS_PATH / "models").mkdir(parents=True, exist_ok=True)
(ARTIFACTS_PATH / "figures").mkdir(parents=True, exist_ok=True)

@dataclass
class ExperimentConfig:
    """Параметры эксперимента"""
    # Данные
    num_classes: int = 112
    image_size: int = 224
    
    # Обучение (этап A: только head)
    stage1_epochs: int = 5
    stage1_lr: float = 1e-3
    
    # Обучение (этап B: fine-tuning)
    stage2_epochs: int = 10
    stage2_lr: float = 1e-5
    
    # Общие параметры
    batch_size: int = 32
    random_seed: int = 42
    
    # Метрики
    metrics: list = None  # ['accuracy', 'f1', 'precision', 'recall']
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']

# Архитектуры для сравнения
ARCHITECTURES = {
    "resnet50": {
        "model_name": "resnet50",
        "params_million": 25.6,
        "pretrained": True
    },
    "efficientnet_b0": {
        "model_name": "efficientnet_b0", 
        "params_million": 5.3,
        "pretrained": True
    },
    "mobilenet_v3": {
        "model_name": "mobilenet_v3_large",
        "params_million": 5.5,
        "pretrained": True
    }
}