# Исходный код проекта

В этой папке размещается **основной код проекта**.


## Описание модулей

| Файл | Назначение |
|------|------------|
| `api.py` | API-сервис на FastAPI. Запускает сервер, обрабатывает запросы, логирует. |
| `model_utils.py` | Класс `PillClassifier`: загрузка модели, preprocess, predict, top-5. |
| `config.py` | Параметры эксперимента: пути, batch_size, num_epochs, архитектуры. |

## Запуск сервиса

```bash
cd ml_Yakunicheva
.venv\Scripts\activate
uvicorn project.src.api:app --host 0.0.0.0 --port 8000
```