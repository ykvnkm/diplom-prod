# RescueAI (локальный запуск)

Короткая инструкция, как поднять сервисы и подключить RTSP поток.

## 1) Сервис детекции (порт 8001)

```bash
python -m uvicorn services.detection_service:app --host 0.0.0.0 --port 8001
```

## 2) Сервис навигации (порт 8000)

```bash
python -m uvicorn services.navigation_service:app --host 0.0.0.0 --port 8000
```

Откройте в браузере: `http://127.0.0.1:8000`

## 3) RTSP сервер (mediamtx)

```bash
mediamtx mediamtx.yml
```

Если `mediamtx` не в PATH, запускайте бинарник из каталога, где он лежит.

## 4) RTSP трансляция через ffmpeg (зацикленное видео)

Пример для локального видео `test_videos/test_2.mp4`:

```bash
ffmpeg -re -stream_loop -1 -i test_videos/test_2.mp4 \
  -c:v libx264 -preset veryfast -tune zerolatency \
  -f rtsp rtsp://127.0.0.1:8554/live
```

Дальше вставьте URL `rtsp://127.0.0.1:8554/live` в UI во вкладке RTSP и запустите обработку.

## Примечания

- Детекция работает отдельным сервисом на `http://127.0.0.1:8001`.
- Навигация/интерфейс на `http://127.0.0.1:8000`.
- Если используете локальные модели (например, DETR или NanoDet), убедитесь, что пути к весам корректны в коде.

## Docker (единый контейнер)

Контейнер поднимает оба сервиса внутри:
- `detection_service` на `8001` (внутри контейнера)
- `unified_navigation_service` на `8010` (наружу публикуется `8010`)

### Быстрый старт

```bash
docker compose -f docker-compose.unified.yml up --build
```

UI: `http://127.0.0.1:8010`

### Что попадает в образ

Сборка использует строгий `.dockerignore` и берёт только нужное:
- `services/`
- `configs/`
- `nanodet/`
- `scripts/docker/`
- docker-файлы

Все датасеты, отчёты, тестовые видео и прочие артефакты не попадают в образ.

### Автоподгрузка моделей и конфигов на другом устройстве

При старте контейнера запускается `scripts/docker/bootstrap_models.sh`:
- если нет `models/yolov8n_baseline_multiscale.pt`, он скачивается автоматически;
- NanoDet ONNX/конфиги берутся из `nanodet/` внутри образа;
- конфиги сервисов уже в образе (`configs/*.yaml`).

По умолчанию для YOLO используется fallback URL (Ultralytics `yolov8n.pt`), чтобы контейнер стартовал без ручных шагов.
Если нужен **ваш** кастомный `yolov8n_baseline_multiscale.pt`, задайте:

```yaml
environment:
  YOLO_WEIGHTS_URL: "https://<your-storage>/yolov8n_baseline_multiscale.pt"
```

(в `docker-compose.unified.yml` или через `-e`).

Для NanoDet теперь базовый режим: **обязательный `.pth`**:

```yaml
environment:
  REQUIRE_NANODET_PTH: "1"
  NANODET_PTH_URL: "https://<your-storage>/nanodet-plus-m-1.5x_416.pth"
```

ONNX теперь опционален (только если хотите запускать через ONNX Runtime):

```yaml
environment:
  REQUIRE_NANODET_ONNX: "1"
  NANODET_ONNX: "1"
  NANODET_ONNX_URL: "https://<your-storage>/nanodet-plus-m-1.5x_416.onnx"
```

Конфиг NanoDet (`nanodet/config/nanodet-plus-m-1.5x_416.yml`) уже включён в образ автоматически.
Отдельно заливать его в URL не нужно, если устраивает стандартный конфиг.
Свой конфиг нужен только при кастомной архитектуре/пайплайне, тогда можно передать `NANODET_CONFIG_URL`.

### Как сделать доступный URL для весов

Вариант 1: GitHub Releases
1. Создайте репозиторий (или используйте существующий).
2. `Releases` -> `Draft a new release`.
3. Прикрепите файл `yolov8n_baseline_multiscale.pt` (и при необходимости NanoDet ONNX/PTH).
4. Опубликуйте релиз.
5. Скопируйте прямую ссылку вида:  
   `https://github.com/<org>/<repo>/releases/download/<tag>/<file>`
6. Подставьте эту ссылку в `YOLO_WEIGHTS_URL` / `NANODET_ONNX_URL`.

Вариант 2: S3/MinIO
1. Загрузите файл в bucket.
2. Сделайте объект публичным **или** выдайте pre-signed URL.
3. Используйте URL в переменных окружения.
4. Для pre-signed URL задайте срок действия достаточно большим (например, на время демо).

### Чтобы не скачивать модель каждый запуск

В `docker-compose.unified.yml` уже есть volume:
- `rescueai_models:/app/models`

Модель кэшируется и повторно используется при следующем запуске контейнера.
