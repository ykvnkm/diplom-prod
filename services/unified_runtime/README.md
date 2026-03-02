# Unified Runtime Services

Новый комплект сервисов с объединенной логикой режимов и UI, перенесенным с `navigation_service.py`.

## Сервисы

1. `unified_navigation_service.py`
- UI/API на базе `navigation_service.py`
- Режимы:
  - `НСУ -> локальный`: `video`, `rtsp`, `frames`
  - `НСУ -> потоковый`: `video`, `rtsp`, `frames` (источник идет с RaspberryPi source-service)
  - `Edge`: заглушка (зарезервировано под следующий этап)
- В НСУ разрешены только модели: `yolov8n_baseline_multiscale`, `nanodet`
- WebSocket: `/ws/process/{session_id}`
- Health: `/health`

2. `rpi_source_service.py`
- Сервис для RaspberryPi: отдает только исходный контент через MJPEG
- `POST /source/start` (`file|rtsp|frames`)
- `GET /source/stream/{session_id}`
- `POST /source/stop/{session_id}`

3. `feature_flags.py`
- Флаги включения/выключения режимов по отдельности (через env)

4. `configs/unified_runtime.yaml`
- Конфиг нового сервиса: URL детектора, путь UI-шаблона, флаги режимов
- Можно задать debug-пресеты запуска (`debug_presets`), например:
  - `nsu_frames_yolov8n_alert_contract: configs/nsu_frames_yolov8n_alert_contract.yaml`
  - `nsu_video_yolov8n_fast: configs/nsu_video_yolov8n_fast.yaml`

## Важное про детекцию

Сервис `unified_navigation_service` использует внешний `services/detection_service.py` (как в старой связке).

Перед запуском unified-сервиса поднимите детектор:

```bash
python -m uvicorn services.detection_service:app --host 0.0.0.0 --port 8001
```

## Запуск

```bash
python -m uvicorn services.unified_runtime.unified_navigation_service:app --host 0.0.0.0 --port 8010
```

## Debug-профиль 24 (отдельный запуск)

- В UI (`НСУ -> local -> frames`) включите тумблер:
  - `Debug preset: nsu_frames_yolov8n_alert_contract`
- Сервис применит параметры из [`configs/nsu_frames_yolov8n_alert_contract.yaml`](configs/nsu_frames_yolov8n_alert_contract.yaml):
  - модель (`yolov8n_baseline_multiscale`)
  - инференс-порог (`infer.conf_min`)
  - ограничение скорости обработки по `dataset.fps` (6 FPS)
  - debug-метрики по sweep из `eval.thresholds` с отбором по `target_recall` и `fp_per_min_target`
- Проверка доступных пресетов:
  - `GET /debug/presets`

## Поток кадров (фиксированный запуск)

- Для режима `НСУ -> local -> frames` выбор модели и debug-настроек отключен в UI.
- Этот режим всегда запускается с конфигом:
  - `configs/nsu_frames_yolov8n_alert_contract.yaml`
- Сравнение с GT и метрики для `frames` считаются автоматически по COCO из этого же конфига:
  - `dataset.coco_gt` (сейчас: `public/annotations/val_from_labels.json`)
- Метрики считаются по контракту alert/episode:
  - `Recall_event = episodes_found / episodes_total`
  - `FP/min = false_alerts_total / (mission_duration_sec / 60)`
  - c параметрами из `alert`/`eval` (`window_sec`, `quorum_k`, `cooldown_sec`, `gap_end_sec`, `gt_gap_end_sec`, `match_tolerance_sec`, `min_detections_per_frame`, `thresholds`, `target_recall`, `fp_per_min_target`)
  - `operator_confirm_delay_sec` в UI-оценке не используется.

На RaspberryPi:

```bash
python -m uvicorn services.unified_runtime.rpi_source_service:app --host 0.0.0.0 --port 9100
```

## НСУ-потоковый режим (миссия в realtime)

В UI:
- `Режим вычислений`: `1 - НСУ`
- `Подрежим НСУ`: `1.2 Потоковый (RPi source)`
- `URL source-сервиса RaspberryPi`: например `http://192.168.1.50:9100`

Доступны параметры потоковой миссии:
- `Mission ID`
- `Target FPS`
- `JPEG quality`
- `Jitter (ms)`
- `Realtime pacing`
- `Drop frame on lag`
- `Макс. длит., сек`

Под капотом:
- RaspberryPi отдает поток через `rpi_source_service` (`/source/start`, `/source/stream/{id}`).
- Инференс и навигация выполняются на НСУ (локально в `unified_navigation_service` + `detection_service`).
- В UI выводятся телеметрии канала: `Поток FPS (RPi)` и `Дроп кадров (RPi)`.

## Локальный RTSP (mediamtx + ffmpeg)

Быстрый локальный стенд для вкладки `RTSP поток`:

```bash
bash scripts/rtsp/start_local_rtsp.sh test_videos/test_1.mp4
```

Проверка потока:

```bash
bash scripts/rtsp/probe_local_rtsp.sh
```

Поведение RTSP в unified runtime:
- в UI отправляются только кадры, где есть детекции;
- частота обработки ограничена до ~`6 FPS` по умолчанию (если preset не задает свой `target_fps`).

Остановка:

```bash
bash scripts/rtsp/stop_local_rtsp.sh
```

RTSP URL для UI:

```text
rtsp://127.0.0.1:8554/live
```
