from __future__ import annotations

import datetime
import os
import random
import time
import uuid
from pathlib import Path
from typing import Dict, Generator, List, Optional

import cv2
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

app = FastAPI(title="RPi Source Service", description="Streams raw content from RaspberryPi")

SESSIONS: Dict[str, Dict] = {}
RPI_VIDEO_DIR = Path(os.getenv("RPI_VIDEO_DIR", "/home/ykvnkm/Documents/test_videos"))
RPI_MISSIONS_DIR = Path(os.getenv("RPI_MISSIONS_DIR", "/home/ykvnkm/Documents/missions"))


class StartSourceRequest(BaseModel):
    mode: str = Field("file", pattern="^(file|rtsp|frames)$")
    source: str
    loop: bool = False
    jpeg_quality: int = 80
    mission_id: str = ""
    realtime: bool = True
    target_fps: float = 0.0
    jitter_ms: int = 0
    drop_if_lag: bool = True
    max_duration_sec: float = 0.0


class StartSourceResponse(BaseModel):
    session_id: str
    mode: str
    stream_url: str
    mission_id: str
    target_fps: float
    realtime: bool


class StopSourceResponse(BaseModel):
    status: str


def _video_files_catalog(root: Path) -> List[Dict[str, str]]:
    if not root.exists() or not root.is_dir():
        return []
    out: List[Dict[str, str]] = []
    allowed_ext = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in allowed_ext:
            continue
        out.append(
            {
                "id": p.stem,
                "name": p.name,
                "path": str(p),
            }
        )
    return out


def _missions_catalog(root: Path) -> List[Dict[str, str]]:
    if not root.exists() or not root.is_dir():
        return []
    out: List[Dict[str, str]] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        images_dir = d / "images"
        ann_dir = d / "annotations"
        if not images_dir.exists() or not images_dir.is_dir():
            continue
        ann_jsons = sorted([p for p in ann_dir.rglob("*.json")]) if ann_dir.exists() and ann_dir.is_dir() else []
        out.append(
            {
                "id": d.name,
                "name": d.name,
                "images_dir": str(images_dir),
                "annotations_json": str(ann_jsons[0]) if ann_jsons else "",
            }
        )
    return out


class FolderFrameReader:
    def __init__(self, folder: Path, loop: bool):
        self.folder = folder
        self.loop = loop
        self.files = self._scan_files()
        self.idx = 0

    def _scan_files(self) -> List[Path]:
        exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
        files: List[Path] = []
        for ext in exts:
            files.extend(self.folder.rglob(ext))
        files = sorted(files)
        return files

    def read(self):
        if not self.files:
            return False, None
        if self.idx >= len(self.files):
            if not self.loop:
                return False, None
            self.idx = 0

        img_path = self.files[self.idx]
        self.idx += 1
        frame = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if frame is None:
            return self.read()
        return True, frame


@app.get("/health")
def health():
    active = sum(1 for s in SESSIONS.values() if not s.get("stop", False))
    sessions_short = []
    for sid, meta in SESSIONS.items():
        sessions_short.append(
            {
                "session_id": sid,
                "mission_id": meta.get("mission_id", ""),
                "mode": meta.get("mode"),
                "stop": bool(meta.get("stop", False)),
                "frames_emitted": int(meta.get("frames_emitted", 0)),
                "frames_dropped": int(meta.get("frames_dropped", 0)),
                "target_fps": float(meta.get("target_fps", 0.0)),
            }
        )
    return {"status": "ok", "active_sessions": active, "sessions": sessions_short}


@app.get("/mission/catalog")
def mission_catalog():
    return {
        "status": "ok",
        "video_root": str(RPI_VIDEO_DIR),
        "missions_root": str(RPI_MISSIONS_DIR),
        "videos": _video_files_catalog(RPI_VIDEO_DIR),
        "missions": _missions_catalog(RPI_MISSIONS_DIR),
    }


@app.get("/source/session/{session_id}")
def source_session_info(session_id: str):
    meta = SESSIONS.get(session_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Session not found")
    started_at = float(meta.get("started_at_monotonic", 0.0) or 0.0)
    now = time.monotonic()
    uptime = max(0.0, now - started_at) if started_at > 0 else 0.0
    emitted = int(meta.get("frames_emitted", 0))
    return {
        "session_id": session_id,
        "mission_id": meta.get("mission_id", ""),
        "mode": meta.get("mode", ""),
        "source": meta.get("source", ""),
        "stop": bool(meta.get("stop", False)),
        "target_fps": float(meta.get("target_fps", 0.0)),
        "realtime": bool(meta.get("realtime", True)),
        "frames_emitted": emitted,
        "frames_dropped": int(meta.get("frames_dropped", 0)),
        "avg_emit_fps": (float(emitted) / uptime) if uptime > 1e-6 else 0.0,
        "uptime_sec": uptime,
    }


@app.post("/source/start", response_model=StartSourceResponse)
def start_source(req: StartSourceRequest):
    source = req.source.strip()
    if not source:
        raise HTTPException(status_code=400, detail="source is required")

    mode = req.mode
    mission_id = req.mission_id.strip() or f"mission_{uuid.uuid4().hex[:10]}"
    if mode in ("file", "frames"):
        p = Path(source)
        if not p.exists():
            raise HTTPException(status_code=400, detail=f"source not found on RPi: {source}")

    session_id = uuid.uuid4().hex
    created = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    target_fps = float(req.target_fps or 0.0)
    if target_fps < 0:
        target_fps = 0.0
    if mode == "file" and target_fps <= 0:
        cap = cv2.VideoCapture(str(source))
        if cap.isOpened():
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if fps > 0:
                target_fps = fps
        cap.release()
    if mode == "frames" and target_fps <= 0:
        target_fps = 6.0

    SESSIONS[session_id] = {
        "session_id": session_id,
        "mission_id": mission_id,
        "mode": mode,
        "source": source,
        "loop": bool(req.loop),
        "created": created,
        "stop": False,
        "jpeg_quality": max(10, min(100, int(req.jpeg_quality))),
        "realtime": bool(req.realtime),
        "target_fps": float(target_fps),
        "jitter_ms": max(0, int(req.jitter_ms)),
        "drop_if_lag": bool(req.drop_if_lag),
        "max_duration_sec": max(0.0, float(req.max_duration_sec or 0.0)),
        "frames_emitted": 0,
        "frames_dropped": 0,
        "started_at_monotonic": 0.0,
        "last_emit_ts": 0.0,
    }
    stream_url = f"/source/stream/{session_id}"
    return StartSourceResponse(
        session_id=session_id,
        mode=mode,
        stream_url=stream_url,
        mission_id=mission_id,
        target_fps=float(target_fps),
        realtime=bool(req.realtime),
    )


@app.post("/source/stop/{session_id}", response_model=StopSourceResponse)
def stop_source(session_id: str):
    meta = SESSIONS.get(session_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Session not found")
    meta["stop"] = True
    return StopSourceResponse(status="stopping")


def _mjpeg_generator(session_id: str) -> Generator[bytes, None, None]:
    meta = SESSIONS.get(session_id)
    if not meta:
        return

    mode = meta["mode"]
    source = meta["source"]
    quality = meta.get("jpeg_quality", 80)
    loop = bool(meta.get("loop", False))
    realtime = bool(meta.get("realtime", True))
    target_fps = float(meta.get("target_fps", 0.0))
    jitter_ms = max(0, int(meta.get("jitter_ms", 0)))
    drop_if_lag = bool(meta.get("drop_if_lag", True))
    max_duration_sec = float(meta.get("max_duration_sec", 0.0))
    start_ts = time.monotonic()
    next_emit_ts = start_ts
    emitted = 0
    meta["started_at_monotonic"] = start_ts

    cap: Optional[cv2.VideoCapture] = None
    folder_reader: Optional[FolderFrameReader] = None

    try:
        if mode in ("file", "rtsp"):
            cap = cv2.VideoCapture(str(source))
            if not cap.isOpened():
                return
        else:
            folder_reader = FolderFrameReader(Path(source), loop=loop)

        while True:
            if meta.get("stop"):
                break
            if max_duration_sec > 0 and (time.monotonic() - start_ts) > max_duration_sec:
                break

            if mode in ("file", "rtsp") and cap is not None:
                ok, frame = cap.read()
                if not ok:
                    if mode == "rtsp":
                        cap.release()
                        cap = cv2.VideoCapture(str(source))
                        if not cap.isOpened():
                            break
                        continue
                    if loop:
                        cap.release()
                        cap = cv2.VideoCapture(str(source))
                        if not cap.isOpened():
                            break
                        continue
                    break
            else:
                assert folder_reader is not None
                ok, frame = folder_reader.read()
                if not ok:
                    break

            if realtime and target_fps > 0:
                now = time.monotonic()
                lag = now - next_emit_ts
                if lag > (1.0 / max(target_fps, 1e-6)) and drop_if_lag:
                    meta["frames_dropped"] = int(meta.get("frames_dropped", 0)) + 1
                    next_emit_ts += (1.0 / target_fps)
                    continue
                sleep_for = next_emit_ts - now
                if jitter_ms > 0:
                    sleep_for += random.uniform(0.0, float(jitter_ms) / 1000.0)
                if sleep_for > 0:
                    time.sleep(sleep_for)

            ok_jpg, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if not ok_jpg:
                continue

            emitted += 1
            meta["frames_emitted"] = emitted
            meta["last_emit_ts"] = time.monotonic()
            if realtime and target_fps > 0:
                next_emit_ts += (1.0 / target_fps)

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            )
    finally:
        if cap is not None:
            cap.release()
        if session_id in SESSIONS:
            SESSIONS[session_id]["stop"] = True


@app.get("/source/stream/{session_id}")
def source_stream(session_id: str):
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")
    return StreamingResponse(_mjpeg_generator(session_id), media_type="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("services.unified_runtime.rpi_source_service:app", host="0.0.0.0", port=9100, reload=False)
