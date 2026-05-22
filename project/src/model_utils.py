import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io

class PillClassifier:
    def __init__(self, model_name='ResNet50', model_path=None, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model_name = model_name
        
        # Создаём архитектуру
        if model_name == 'ResNet50':
            self.model = models.resnet50(pretrained=False)
            self.model.fc = nn.Linear(2048, 112)
        elif model_name == 'EfficientNet_B0':
            self.model = models.efficientnet_b0(pretrained=False)
            self.model.classifier[1] = nn.Linear(1280, 112)
        elif model_name == 'MobileNet_V3':
            self.model = models.mobilenet_v3_large(pretrained=False)
            self.model.classifier[3] = nn.Linear(1280, 112)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Загружаем веса
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Трансформации для изображения
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Загружаем названия классов
        self.class_names = self._load_class_names()
    
    def _load_class_names(self):
        # Загружаем из CSV или создаём заглушку
        import pandas as pd
        import os
        csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'ogyeiv2', 'extracted_sentences.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, header=None)
            return dict(zip(range(len(df)), df[0].tolist()))
        else:
            return {i: f"Class_{i}" for i in range(112)}
    
    def predict(self, image_bytes):
        """Принимает байты изображения, возвращает класс и вероятность"""
        # Открываем изображение
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Преобразуем
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Предсказание
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        class_id = predicted.item()
        class_name = self.class_names.get(class_id, f"Class_{class_id}")
        confidence_score = confidence.item()
        
        return {
            'class_id': class_id,
            'class_name': class_name,
            'confidence': round(confidence_score, 4),
            'top_5': self._get_top_5(probabilities[0])
        }
    
    def _get_top_5(self, probs):
        """Возвращает топ-5 предсказаний"""
        top5_probs, top5_ids = torch.topk(probs, 5)
        return [
            {'class_id': int(top5_ids[i]), 
             'class_name': self.class_names.get(int(top5_ids[i]), f"Class_{top5_ids[i]}"),
             'confidence': round(top5_probs[i].item(), 4)}
            for i in range(5)
        ]