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

3. Создайте локальный `.env` из шаблона:

```bash
cp .env.example .env
```

4. При необходимости отредактируйте `.env` под ваше окружение (IP RaspberryPi, пути и URL весов).

5. Запустите сервис:

```bash
docker compose -f docker-compose.unified.yml up --build
```

6. Дождитесь логов без ошибок и сообщений о запуске `uvicorn`.

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

## Marker Navigation Matcher Backend

The marker navigation pipeline still uses the red square as the metric reference and keeps the existing homography, scale, smoothing, reset, re-detection, and report generation logic. Only the frame-to-frame correspondence frontend is selectable:

```bash
NAV_MATCHER=legacy
NAV_MATCHER=xfeat_lighterglue
```

`legacy` is the default OpenCV GFTT + pyramidal LK tracker. `xfeat_lighterglue` uses the official `verlab/accelerated_features` XFeat + LighterGlue path (`modules.xfeat.XFeat` + official `LighterGlue`, backed by Kornia LightGlue). The project calls this backend `xfeat_lighterglue` because the official repository documents the smaller trained LightGlue variant as "LighterGlue".

By default XFeat runs in conservative assist mode:

```bash
XFEAT_INTEGRATION_MODE=assist
```

In this mode legacy LK remains the short-baseline tracker and XFeat/LighterGlue is used as a guarded recovery/relocalization frontend when LK cannot provide enough clean correspondences. To force full XFeat replacement for debugging, use:

```bash
XFEAT_INTEGRATION_MODE=replace
```

Local setup:

```bash
pip install -r requirements.txt
python scripts/bootstrap_accelerated_features.py
NAV_MATCHER=xfeat_lighterglue python -m uvicorn services.unified_runtime.unified_navigation_service:app --host 0.0.0.0 --port 8010
```

Docker setup:

```bash
NAV_MATCHER=xfeat_lighterglue docker compose -f docker-compose.unified.yml up --build
```

Expected outputs are unchanged: WebSocket trajectory points, optional processed video or alert frames, and the report zip with `trajectory.csv`, `trajectory_3d.png`, `topdown_xy.png`, and `height_over_time.png` under `runtime/unified/reports`. If `third_party/accelerated_features` is missing and `XFEAT_AUTO_BOOTSTRAP=1`, the service will try to clone the official repo automatically when `xfeat_lighterglue` is first used; set `XFEAT_REPO_DIR=/path/to/accelerated_features` to use a pre-cloned copy.

Useful knobs:

```bash
XFEAT_INTEGRATION_MODE=assist
XFEAT_TOP_K=800
XFEAT_LIGHTERGLUE_MIN_CONF=0.25
XFEAT_MAX_MATCHES=300
XFEAT_MOTION_GATE_PX=35
XFEAT_RANSAC_THR=2.0
XFEAT_ASSIST_MIN_LEGACY_PTS=0
XFEAT_AUTO_BOOTSTRAP=1
XFEAT_REPO_DIR=
XFEAT_GIT_REF=main
```

Matcher diagnostics:

```bash
NAV_MATCHER_DEBUG=1
NAV_MATCHER_DEBUG_EVERY_N=30
NAV_MATCHER_DEBUG_MAX_PAIRS=4
```

Diagnostics are written to `runtime/unified/reports/<report_stem>_matcher_diagnostics` and included in the report zip. The first sampled pair gets `legacy_matches_overlay.png`, `xfeat_matches_overlay.png`, `legacy_inliers_overlay.png`, and `xfeat_inliers_overlay.png`; additional pairs get a frame-number suffix. Metrics are saved in `matcher_metrics.csv` and summarized in `comparison_report.json`.

Known limitations: the first XFeat run may download the official repo and model weights, so it needs network access unless `XFEAT_REPO_DIR` and cached weights are already present. The XFeat backend is only a correspondence frontend for the existing marker pipeline; it does not enable a new monocular VO or recoverPose-only path.
