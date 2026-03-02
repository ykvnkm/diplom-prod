#!/usr/bin/env bash
set -euo pipefail

/app/scripts/docker/bootstrap_models.sh

python -m uvicorn services.detection_service:app --host 0.0.0.0 --port 8001 &
DET_PID=$!
python -m uvicorn services.unified_runtime.unified_navigation_service:app --host 0.0.0.0 --port 8010 &
NAV_PID=$!

cleanup() {
  kill "$DET_PID" "$NAV_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait -n "$DET_PID" "$NAV_PID"
