from __future__ import annotations

import datetime
import uuid
from pathlib import Path
from typing import Dict, Generator, List, Optional

import cv2
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

app = FastAPI(title="RPi Source Service", description="Streams raw content from RaspberryPi")

SESSIONS: Dict[str, Dict] = {}


class StartSourceRequest(BaseModel):
    mode: str = Field("file", pattern="^(file|rtsp|frames)$")
    source: str
    loop: bool = False
    jpeg_quality: int = 80


class StartSourceResponse(BaseModel):
    session_id: str
    mode: str
    stream_url: str


class StopSourceResponse(BaseModel):
    status: str


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
            files.extend(self.folder.glob(ext))
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
    return {"status": "ok", "active_sessions": active}


@app.post("/source/start", response_model=StartSourceResponse)
def start_source(req: StartSourceRequest):
    source = req.source.strip()
    if not source:
        raise HTTPException(status_code=400, detail="source is required")

    mode = req.mode
    if mode in ("file", "frames"):
        p = Path(source)
        if not p.exists():
            raise HTTPException(status_code=400, detail=f"source not found on RPi: {source}")

    session_id = uuid.uuid4().hex
    created = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    SESSIONS[session_id] = {
        "session_id": session_id,
        "mode": mode,
        "source": source,
        "loop": bool(req.loop),
        "created": created,
        "stop": False,
        "jpeg_quality": max(10, min(100, int(req.jpeg_quality))),
    }
    stream_url = f"/source/stream/{session_id}"
    return StartSourceResponse(session_id=session_id, mode=mode, stream_url=stream_url)


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

            ok_jpg, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if not ok_jpg:
                continue

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
