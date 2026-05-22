# Конфигурационные файлы

В этой папке хранятся настройки проекта, вынесенные из кода для удобства изменения.


## Файл `.env.example`

Шаблон переменных окружения для запуска API-сервиса:

```bash
# Модель для API (ResNet50, EfficientNet_B0, MobileNet_V3)
MODEL_NAME=ResNet50

# Порт для сервиса
PORT=8000

# Уровень логирования (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL=INFO