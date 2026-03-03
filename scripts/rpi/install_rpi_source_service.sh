#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <project_dir> <user> <group>"
  echo "Example: $0 /home/ykvnkm/Diplom-prod ykvnkm ykvnkm"
  exit 1
fi

PROJECT_DIR="$1"
RUN_USER="$2"
RUN_GROUP="$3"

SERVICE_NAME="rescueai-rpi-source.service"
SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}"

cat > /tmp/${SERVICE_NAME} <<EOF
[Unit]
Description=RescueAI RPi Source Service
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${RUN_USER}
Group=${RUN_GROUP}
WorkingDirectory=${PROJECT_DIR}
Environment=PYTHONUNBUFFERED=1
Environment=RPI_VIDEO_DIR=/home/${RUN_USER}/Documents/test_videos
Environment=RPI_MISSIONS_DIR=/home/${RUN_USER}/Documents/missions
Environment=RPI_RTSP_ENABLE=1
Environment=RPI_RTSP_PORT=8554
Environment=RPI_RTSP_PATH_PREFIX=live
ExecStart=${PROJECT_DIR}/.venv/bin/python -m uvicorn services.unified_runtime.rpi_source_service:app --host 0.0.0.0 --port 9100
Restart=always
RestartSec=2

[Install]
WantedBy=multi-user.target
EOF

sudo mv /tmp/${SERVICE_NAME} "${SERVICE_PATH}"
sudo systemctl daemon-reload
sudo systemctl enable "${SERVICE_NAME}"
sudo systemctl restart "${SERVICE_NAME}"

echo "Installed and started ${SERVICE_NAME}"
sudo systemctl status "${SERVICE_NAME}" --no-pager -l || true
