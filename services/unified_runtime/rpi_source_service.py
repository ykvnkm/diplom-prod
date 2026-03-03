from __future__ import annotations

import datetime
import os
import random
import shutil
import subprocess
import time
import uuid
import tempfile
from pathlib import Path
from typing import Dict, Generator, List, Optional

import cv2
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

app = FastAPI(title="RPi Source Service", description="Streams raw content from RaspberryPi")


def _load_env_file() -> None:
    # Local fallback for runs without systemd/docker env injection.
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        val = value.strip()
        if len(val) >= 2 and ((val[0] == val[-1] == '"') or (val[0] == val[-1] == "'")):
            val = val[1:-1]
        os.environ[key] = val


_load_env_file()


SESSIONS: Dict[str, Dict] = {}
RPI_VIDEO_DIR = Path(os.getenv("RPI_VIDEO_DIR", "/home/pi/Documents/test_videos"))
RPI_MISSIONS_DIR = Path(os.getenv("RPI_MISSIONS_DIR", "/home/pi/Documents/missions"))
RPI_RTSP_HOST = os.getenv("RPI_RTSP_HOST", "").strip()
RPI_RTSP_PORT = int(os.getenv("RPI_RTSP_PORT", "8554"))
RPI_RTSP_PATH_PREFIX = os.getenv("RPI_RTSP_PATH_PREFIX", "live").strip().strip("/")
RPI_RTSP_ENABLE = str(os.getenv("RPI_RTSP_ENABLE", "1")).strip().lower() in ("1", "true", "yes", "on")


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
    rtsp_url: str = ""
    backend: str = "mjpeg"
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


def _is_under_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def _is_allowed_source_file(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    return _is_under_root(path, RPI_VIDEO_DIR) or _is_under_root(path, RPI_MISSIONS_DIR)


def _resolve_rtsp_host(host_header: str) -> str:
    if RPI_RTSP_HOST:
        return RPI_RTSP_HOST
    host = str(host_header or "").split(":")[0].strip()
    return host or "127.0.0.1"


def _ffmpeg_path() -> str:
    return shutil.which("ffmpeg") or ""


def _frames_glob_pattern(folder: str) -> str:
    root = Path(folder)
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    for pat in patterns:
        # Для миссий с вложенными папками используем рекурсивный glob-паттерн.
        if any(root.rglob(pat)):
            return str(root / "**" / pat)
    return str(root / "**" / "*.jpg")


def _scan_frame_files(folder: str) -> List[Path]:
    root = Path(folder)
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    files: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix in exts:
            files.append(p)
    return sorted(files)


def _build_frames_concat_file(session_id: str, folder: str, fps: float) -> Optional[Path]:
    files = _scan_frame_files(folder)
    if not files:
        return None
    dur = 1.0 / max(float(fps), 1e-6)
    lines: List[str] = []
    for p in files:
        safe = str(p).replace("'", "'\\''")
        lines.append(f"file '{safe}'\n")
        lines.append(f"duration {dur:.6f}\n")
    safe_last = str(files[-1]).replace("'", "'\\''")
    lines.append(f"file '{safe_last}'\n")
    out = Path(tempfile.gettempdir()) / f"rpi_frames_concat_{session_id}.txt"
    out.write_text("".join(lines), encoding="utf-8")
    return out


def _rtsp_publish_url(host_header: str, session_id: str) -> tuple[str, str]:
    _ = session_id
    rtsp_host = _resolve_rtsp_host(host_header)
    # Используем фиксированный путь (по умолчанию /live), чтобы быть совместимыми
    # с mediamtx-конфигами, где явно описан конкретный path.
    stream_key = RPI_RTSP_PATH_PREFIX or "live"
    public_url = f"rtsp://{rtsp_host}:{RPI_RTSP_PORT}/{stream_key}"
    # ffmpeg публикует в локальный mediamtx на этом же устройстве.
    ingest_url = f"rtsp://127.0.0.1:{RPI_RTSP_PORT}/{stream_key}"
    return public_url, ingest_url


def _start_rtsp_publisher(meta: Dict, host_header: str) -> tuple[Optional[subprocess.Popen], str]:
    if not RPI_RTSP_ENABLE:
        return None, ""
    ffmpeg = _ffmpeg_path()
    if not ffmpeg:
        return None, ""
    mode = str(meta.get("mode", ""))
    source = str(meta.get("source", ""))
    loop = bool(meta.get("loop", False))
    target_fps = float(meta.get("target_fps", 0.0))
    if target_fps <= 0:
        target_fps = 6.0 if mode == "frames" else 10.0
    target_fps = min(target_fps, 6.0)
    public_url, ingest_url = _rtsp_publish_url(host_header, str(meta.get("session_id", "")))
    common_out = [
        "-an",
        "-vf",
        "scale=640:-2:flags=fast_bilinear",
        "-r",
        f"{target_fps:.3f}",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-tune",
        "zerolatency",
        "-profile:v",
        "baseline",
        "-pix_fmt",
        "yuv420p",
        "-bf",
        "0",
        "-b:v",
        "700k",
        "-maxrate",
        "900k",
        "-bufsize",
        "1800k",
        "-g",
        str(max(12, int(round(target_fps * 2)))),
        "-keyint_min",
        str(max(6, int(round(target_fps)))),
        "-f",
        "rtsp",
        "-rtsp_transport",
        "tcp",
        ingest_url,
    ]
    if mode == "file":
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-re"]
        if loop:
            cmd += ["-stream_loop", "-1"]
        cmd += ["-i", source] + common_out
    elif mode == "frames":
        concat_file = _build_frames_concat_file(str(meta.get("session_id", "")), source, target_fps)
        if concat_file is None or not concat_file.exists():
            meta["publisher_error"] = f"В папке не найдены кадры: {source}"
            return None, ""
        meta["frames_concat_file"] = str(concat_file)
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-re",
            "-f",
            "concat",
            "-safe",
            "0",
        ]
        if loop:
            cmd += ["-stream_loop", "-1"]
        cmd += [
            "-i",
            str(concat_file),
        ] + common_out
    elif mode == "rtsp":
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-rtsp_transport",
            "tcp",
            "-i",
            source,
        ] + common_out
    else:
        return None, ""
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)  # noqa: S603
    time.sleep(0.6)
    if proc.poll() is not None:
        err = ""
        if proc.stderr is not None:
            try:
                raw = proc.stderr.read() or b""
                err = raw.decode("utf-8", errors="ignore").strip()[-500:]
            except Exception:
                err = ""
        meta["publisher_error"] = err or "ffmpeg publisher exited unexpectedly"
        return None, ""
    meta["publisher_error"] = ""
    return proc, public_url


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
                "backend": str(meta.get("backend", "mjpeg")),
                "rtsp_url": str(meta.get("rtsp_url", "")),
                "publisher_running": bool(meta.get("publisher_running", False)),
                "publisher_error": str(meta.get("publisher_error", "")),
            }
        )
    return {
        "status": "ok",
        "active_sessions": active,
        "rtsp_enabled": bool(RPI_RTSP_ENABLE and bool(_ffmpeg_path())),
        "sessions": sessions_short,
    }


@app.get("/mission/catalog")
def mission_catalog():
    rtsp_ready = bool(RPI_RTSP_ENABLE and bool(_ffmpeg_path()))
    return {
        "status": "ok",
        "video_root": str(RPI_VIDEO_DIR),
        "missions_root": str(RPI_MISSIONS_DIR),
        "rtsp_enabled": rtsp_ready,
        "rtsp_host": (RPI_RTSP_HOST or ""),
        "rtsp_port": int(RPI_RTSP_PORT),
        "rtsp_path_prefix": RPI_RTSP_PATH_PREFIX,
        "videos": _video_files_catalog(RPI_VIDEO_DIR),
        "missions": _missions_catalog(RPI_MISSIONS_DIR),
    }


@app.get("/source/session/{session_id}")
def source_session_info(session_id: str):
    meta = SESSIONS.get(session_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Session not found")
    proc = meta.get("publisher_proc")
    if proc is not None and proc.poll() is not None:
        meta["publisher_running"] = False
        meta["publisher_proc"] = None
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
        "backend": str(meta.get("backend", "mjpeg")),
        "rtsp_url": str(meta.get("rtsp_url", "")),
        "publisher_running": bool(meta.get("publisher_running", False)),
        "publisher_error": str(meta.get("publisher_error", "")),
    }


@app.get("/source/raw_file")
def source_raw_file(path: str):
    src = Path(str(path or "").strip())
    if not src.is_absolute():
        raise HTTPException(status_code=400, detail="path должен быть абсолютным путем на RPi")
    if not _is_allowed_source_file(src):
        raise HTTPException(status_code=403, detail="Запрошенный файл недоступен")
    return FileResponse(str(src))


@app.post("/source/start", response_model=StartSourceResponse)
def start_source(req: StartSourceRequest, request: Request):
    source = req.source.strip()
    if not source:
        raise HTTPException(status_code=400, detail="source is required")

    mode = req.mode
    mission_id = req.mission_id.strip() or f"mission_{uuid.uuid4().hex[:10]}"

    # Один активный RTSP-паблишер: перед стартом новой миссии гасим предыдущие.
    for sid, smeta in list(SESSIONS.items()):
        if bool(smeta.get("stop", False)):
            continue
        proc = smeta.get("publisher_proc")
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            smeta["publisher_running"] = False
            smeta["publisher_proc"] = None
        smeta["stop"] = True
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
        "backend": "mjpeg",
        "rtsp_url": "",
        "publisher_running": False,
        "publisher_error": "",
        "publisher_proc": None,
    }
    rtsp_proc, rtsp_url = _start_rtsp_publisher(SESSIONS[session_id], request.headers.get("host", ""))
    if rtsp_proc is not None and rtsp_url:
        SESSIONS[session_id]["publisher_proc"] = rtsp_proc
        SESSIONS[session_id]["publisher_running"] = True
        SESSIONS[session_id]["rtsp_url"] = rtsp_url
        SESSIONS[session_id]["backend"] = "rtsp"
    else:
        err = str(SESSIONS[session_id].get("publisher_error", "")).strip() or "RTSP publisher не запустился"
        # В потоковом контуре ожидается только RTSP.
        raise HTTPException(status_code=500, detail=err)
    stream_url = f"/source/stream/{session_id}"
    return StartSourceResponse(
        session_id=session_id,
        mode=mode,
        stream_url=stream_url,
        rtsp_url=SESSIONS[session_id].get("rtsp_url", ""),
        backend=SESSIONS[session_id].get("backend", "mjpeg"),
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
    proc = meta.get("publisher_proc")
    if proc is not None:
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        meta["publisher_running"] = False
        meta["publisher_proc"] = None
    concat_file = meta.get("frames_concat_file")
    if concat_file:
        try:
            Path(str(concat_file)).unlink(missing_ok=True)
        except Exception:
            pass
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
                    # Не уходим в бесконечный drop: выравниваем расписание и отправляем текущий кадр.
                    meta["frames_dropped"] = int(meta.get("frames_dropped", 0)) + 1
                    next_emit_ts = now
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
        concat_file = meta.get("frames_concat_file")
        if concat_file:
            try:
                Path(str(concat_file)).unlink(missing_ok=True)
            except Exception:
                pass
        proc = meta.get("publisher_proc")
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            meta["publisher_running"] = False
            meta["publisher_proc"] = None
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
