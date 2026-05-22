from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import os
import time
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Пути к моделям (можно через переменные окружения)
MODEL_PATHS = {
    'ResNet50': '../artifacts/models/ResNet50_best.pt',
    'EfficientNet_B0': '../artifacts/models/EfficientNet_B0_best.pt',
    'MobileNet_V3': '../artifacts/models/MobileNet_V3_best.pt'
}

# Создаём приложение
app = FastAPI(
    title="Pill Classification API",
    description="API для распознавания таблеток по изображению (112 классов)",
    version="1.0.0"
)

# Глобальная переменная для модели (загружается при старте)
classifier = None
current_model_name = None

@app.on_event("startup")
async def load_model():
    """Загружаем модель при старте сервера"""
    global classifier, current_model_name
    
    # Используем лучшую модель (ResNet50)
    model_name = os.getenv('MODEL_NAME', 'ResNet50')
    model_path = MODEL_PATHS.get(model_name)
    
    if not model_path or not os.path.exists(model_path):
        logger.warning(f"Model {model_name} not found, using mock")
        return
    
    from model_utils import PillClassifier
    classifier = PillClassifier(model_name=model_name, model_path=model_path)
    current_model_name = model_name
    logger.info(f"✅ Model {model_name} loaded successfully")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": classifier is not None,
        "model_name": current_model_name,
        "cuda_available": torch.cuda.is_available() if classifier else None
    }

@app.get("/info")
async def model_info():
    """Информация о модели"""
    if not classifier:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": current_model_name,
        "num_classes": 112,
        "class_names_sample": list(classifier.class_names.items())[:10]
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Предсказание класса таблетки по изображению"""
    start_time = time.time()
    
    # Проверяем тип файла
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Читаем файл
    try:
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")
    
    # Предсказание
    try:
        if not classifier:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        result = classifier.predict(image_bytes)
        latency_ms = (time.time() - start_time) * 1000
        
        return JSONResponse({
            **result,
            "latency_ms": round(latency_ms, 2),
            "model_used": current_model_name
        })
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/compare")
async def compare_models(file: UploadFile = File(...)):
    """Сравнение всех моделей на одном изображении"""
    image_bytes = await file.read()
    results = {}
    
    for model_name, model_path in MODEL_PATHS.items():
        if os.path.exists(model_path):
            from model_utils import PillClassifier
            clf = PillClassifier(model_name=model_name, model_path=model_path)
            results[model_name] = clf.predict(image_bytes)
    
    return JSONResponse(results)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)