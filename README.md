# RescueAI: запуск через Docker

## Для тех, кто заходит в репозиторий впервые

1. Установите Docker Desktop.
2. Запустите Docker Desktop и дождитесь статуса `Engine running`.
3. Откройте Терминал.

## Клонирование и запуск

1. Клонируйте репозиторий:

```bash
git clone https://github.com/ykvnkm/diplom-prod.git
```

2. Перейдите в папку проекта:

```bash
cd diplom-prod
```

3. Запустите сервис:

```bash
docker compose -f docker-compose.unified.yml up --build
```

4. Дождитесь логов без ошибок и сообщений о запуске `uvicorn`.

## Куда заходить

Откройте в браузере:

`http://127.0.0.1:8010`

## Как остановить

1. Вернитесь в окно терминала, где запущен сервис.
2. Нажмите `Ctrl + C`.

Чтобы убрать контейнеры и сеть:

```bash
docker compose -f docker-compose.unified.yml down
```

## Локальный запуск без Docker

1. Установите зависимости:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Запустите сервисы:

```bash
python -m uvicorn services.detection_service:app --host 0.0.0.0 --port 8001
python -m uvicorn services.unified_runtime.unified_navigation_service:app --host 0.0.0.0 --port 8010
```

3. В `detection_service` добавлена автоподгрузка весов, если локального файла нет.
Используются те же env-переменные, что и в `docker-compose.unified.yml`:
- `YOLO_WEIGHTS_PATH`, `YOLO_WEIGHTS_URL`
- `NANODET_WEIGHTS`, `NANODET_PTH_URL`
- `NANODET_ONNX_PATH`, `NANODET_ONNX_URL`, `REQUIRE_NANODET_ONNX`
- `NANODET_CONFIG`, `NANODET_CONFIG_URL`, `REQUIRE_NANODET_CONFIG`
- `REQUIRE_NANODET_PTH`

Если файл существует локально, скачивания не будет.
