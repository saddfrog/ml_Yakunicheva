## HTTP API

Сервер запускается: `uv run uvicorn eda_cli.api:app --reload --port 8000`

### Эндпоинты:
- `GET /health` - проверка работы сервиса
- `POST /quality` - оценка качества по метаданным
- `POST /quality-from-csv` - оценка качества из CSV файла
- `POST /quality-flags-from-csv` - **НОВЫЙ** полный набор флагов качества (включая эвристики из HW03)

### Пример:
```bash
curl -X POST -F "file=@data/example.csv" http://localhost:8000/quality-flags-from-csv