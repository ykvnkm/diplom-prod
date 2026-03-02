from __future__ import annotations

import os
from typing import Dict


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


MODE_FLAGS: Dict[str, bool] = {
    "enable_nsu": _env_bool("UNIFIED_ENABLE_NSU", True),
    "enable_nsu_local": _env_bool("UNIFIED_ENABLE_NSU_LOCAL", True),
    "enable_nsu_stream": _env_bool("UNIFIED_ENABLE_NSU_STREAM", True),
    "enable_nsu_local_video": _env_bool("UNIFIED_ENABLE_NSU_LOCAL_VIDEO", True),
    "enable_nsu_local_rtsp": _env_bool("UNIFIED_ENABLE_NSU_LOCAL_RTSP", True),
    "enable_nsu_local_frames": _env_bool("UNIFIED_ENABLE_NSU_LOCAL_FRAMES", True),
    "enable_nsu_stream_video": _env_bool("UNIFIED_ENABLE_NSU_STREAM_VIDEO", True),
    "enable_nsu_stream_rtsp": _env_bool("UNIFIED_ENABLE_NSU_STREAM_RTSP", True),
    "enable_nsu_stream_frames": _env_bool("UNIFIED_ENABLE_NSU_STREAM_FRAMES", True),
    "enable_edge": _env_bool("UNIFIED_ENABLE_EDGE", True),
}


def is_enabled(flag: str) -> bool:
    return MODE_FLAGS.get(flag, False)
