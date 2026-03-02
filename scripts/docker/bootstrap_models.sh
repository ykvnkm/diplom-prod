#!/usr/bin/env bash
set -euo pipefail

mkdir -p /app/models /app/runtime/unified/uploads /app/runtime/unified/reports

FRAMES_DIR="${NSU_LOCAL_FRAMES_DIR:-/app/public/images}"
COCO_PATH="${NSU_LOCAL_FRAMES_COCO:-/app/public/annotations/val_from_labels.json}"
mkdir -p "$FRAMES_DIR" "$(dirname "$COCO_PATH")"

fetch_if_missing() {
  local target="$1"
  local url="$2"
  mkdir -p "$(dirname "$target")"

  if [[ -f "$target" ]]; then
    echo "[bootstrap] exists: $target"
    return 0
  fi

  if [[ -z "$url" ]]; then
    echo "[bootstrap] missing file and url not provided: $target"
    return 1
  fi

  echo "[bootstrap] downloading: $target"
  curl -fL --retry 5 --retry-delay 2 --connect-timeout 15 "$url" -o "$target.tmp"
  mv "$target.tmp" "$target"
}

YOLO_TARGET="${YOLO_WEIGHTS_PATH:-/app/models/yolov8n_baseline_multiscale.pt}"
YOLO_URL="${YOLO_WEIGHTS_URL:-https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt}"
fetch_if_missing "$YOLO_TARGET" "$YOLO_URL"

if [[ "${REQUIRE_NANODET_CONFIG:-0}" == "1" ]]; then
  NANODET_CONFIG_TARGET="${NANODET_CONFIG:-/app/nanodet/config/nanodet-plus-m-1.5x_416.yml}"
  NANODET_CONFIG_URL="${NANODET_CONFIG_URL:-}"
  fetch_if_missing "$NANODET_CONFIG_TARGET" "$NANODET_CONFIG_URL"
fi

if [[ "${REQUIRE_NANODET_ONNX:-0}" == "1" ]]; then
  NANODET_ONNX_TARGET="${NANODET_ONNX_PATH:-/app/nanodet/nanodet/model/weights/nanodet-plus-m-1.5x_416.onnx}"
  NANODET_ONNX_URL="${NANODET_ONNX_URL:-}"
  fetch_if_missing "$NANODET_ONNX_TARGET" "$NANODET_ONNX_URL"
fi

if [[ "${REQUIRE_NANODET_PTH:-1}" == "1" ]]; then
  NANODET_PTH_TARGET="${NANODET_WEIGHTS:-/app/nanodet/nanodet/model/weights/nanodet-plus-m-1.5x_416.pth}"
  NANODET_PTH_URL="${NANODET_PTH_URL:-}"
  fetch_if_missing "$NANODET_PTH_TARGET" "$NANODET_PTH_URL"
fi

echo "[bootstrap] model bootstrap completed"
