FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DET_DEVICE=cpu \
    UNIFIED_DETECTION_URL=http://127.0.0.1:8001 \
    REQUIRE_NANODET_PTH=1 \
    REQUIRE_NANODET_ONNX=0 \
    NANODET_ONNX=0 \
    NANODET_CONFIG=/app/nanodet/config/nanodet-plus-m-1.5x_416.yml \
    NANODET_WEIGHTS=/app/nanodet/nanodet/model/weights/nanodet-plus-m-1.5x_416.pth \
    NANODET_ONNX_PATH=/app/nanodet/nanodet/model/weights/nanodet-plus-m-1.5x_416.onnx

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY docker/unified-runtime/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY services /app/services
COPY configs /app/configs
COPY nanodet /app/nanodet
COPY scripts/docker /app/scripts/docker

RUN chmod +x /app/scripts/docker/*.sh \
    && mkdir -p /app/models /app/runtime/unified/uploads /app/runtime/unified/reports /app/public/images /app/public/annotations

EXPOSE 8010 8001

ENTRYPOINT ["/app/scripts/docker/start_unified_stack.sh"]
