from __future__ import annotations

import asyncio
import base64
import datetime
import io
import json
import os
import time
import uuid
import zipfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import cv2
import matplotlib
import numpy as np
import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect  # type: ignore
from fastapi.responses import FileResponse, HTMLResponse  # type: ignore

from services.unified_runtime.feature_flags import MODE_FLAGS, is_enabled

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


UPLOAD_DIR = Path("runtime/unified/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = Path("runtime/unified/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)
DETECTION_URL = "http://127.0.0.1:8001"
NAV_W = 853
NAV_H = 480
FX_NAV = 1100
FY_NAV = 1100
CX_NAV = NAV_W / 2
CY_NAV = NAV_H / 2
K_NAV = np.array([[FX_NAV, 0, CX_NAV], [0, FY_NAV, CY_NAV], [0, 0, 1]], dtype=np.float32)
MARKER_3D = np.array([[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0]], dtype=np.float32)
PX_TO_M_XY = 0.15
PX_TO_M_Z = 0.02
MAX_XY_STEP = 1.5
MAX_DH = 0.4
XY_ALPHA = 0.15
Z_ALPHA = 0.25
MARKER_ALPHA = 0.3
PX_TO_M_Z_ALT = 0.02
MAX_DH_ALT = 0.4
Z_ALPHA_ALT = 0.25
MARKER_ALPHA_ALT = 0.3
SMOOTH_WINDOW = 5
SMOOTH_LR_XY = 0.15
SMOOTH_LR_Z = 0.25
SMOOTH_JUMP_XY = 0.7
SMOOTH_JUMP_Z = 0.3
AUTO_MARKER_SECONDS = 3.0
ROI_TOP_RATIO = 0.45
MAX_CORNERS_MARKER = 800
MIN_TRACK_PTS_MARKER = 120
GFTT_QUALITY = 0.01
GFTT_MIN_DIST = 12
GFTT_BLOCK = 7
LK_WIN_MARKER = 31
LK_LEVELS_MARKER = 4
REDETECT_MIN_PTS_MARKER = 0
MAX_STEP_SCALE_MARKER = 1.3
RANSAC_THR_MARKER = 3.0
MIN_INLIERS_MARKER = 140
MIN_INLIER_RATIO_MARKER = 0.35
HARD_RESET_RATIO_MARKER = 0.25
FB_THR_MARKER = 1.5
LK_ERR_THR_MARKER = 25.0
MAX_SPEED_MARKER = 9.0
MARKER_CHECK_EVERY = 45
MARKER_AREA_MIN = 3500.0
FLIP_X_MARKER = False
FLIP_Y_MARKER = True
MARKER_SIZE_XY_M = 2.0
MARKER_RESIZE_W = 960
MARKER_RESIZE_H = 540
CONFIG_PATH = Path("configs/unified_runtime.yaml")
UI_TEMPLATE_PATH = Path("services/unified_runtime/templates/unified_navigation_ui.html")
NSU_LOCAL_FRAMES_DIR = Path(os.getenv("NSU_LOCAL_FRAMES_DIR", "public/images"))
NSU_LOCAL_FRAMES_COCO = Path(os.getenv("NSU_LOCAL_FRAMES_COCO", "public/annotations/val_from_labels.json"))
NSU_LOCAL_VIDEO_PRESET = "nsu_video_yolov8n_fast"

ALLOWED_NSU_MODELS = {
    "yolov8n_baseline_multiscale": "yolov8n_baseline_multiscale",
    "nanodet": "nanodet15",
}
DEBUG_PRESET_PATHS = {
    "nsu_frames_yolov8n_alert_contract": Path("configs/nsu_frames_yolov8n_alert_contract.yaml"),
    "nsu_video_yolov8n_fast": Path("configs/nsu_video_yolov8n_fast.yaml"),
}


@dataclass
class SourceProfile:
    detection_stride: int
    emit_only_detections: bool


def source_profile(kind: str) -> SourceProfile:
    if kind == "rtsp":
        return SourceProfile(detection_stride=3, emit_only_detections=True)
    if kind == "frames":
        return SourceProfile(detection_stride=1, emit_only_detections=True)
    return SourceProfile(detection_stride=1, emit_only_detections=False)


class DetectionClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def health(self) -> Dict:
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            resp.raise_for_status()
            payload = resp.json()
            payload["_reachable"] = True
            return payload
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "_reachable": False, "detail": str(exc)}

    def detect(
        self,
        frame_bgr: np.ndarray,
        model: str,
        conf: float | None = None,
        iou: float | None = None,
        max_det: int | None = None,
    ):
        ok, buf = cv2.imencode(".jpg", frame_bgr)
        if not ok:
            return [], 0, 0.0
        try:
            payload = {"model": model}
            if conf is not None:
                payload["conf"] = str(conf)
            if iou is not None:
                payload["iou"] = str(iou)
            if max_det is not None:
                payload["max_det"] = str(max_det)
            resp = requests.post(
                f"{self.base_url}/detect",
                files={"file": ("frame.jpg", buf.tobytes(), "image/jpeg")},
                data=payload,
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            detections = data.get("detections", [])
            count = int(data.get("person_count", len(detections)))
            latency = float(data.get("inference_ms", 0.0))
            return detections, count, latency
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] detection request failed: {exc}")
            return [], 0, 0.0


def _apply_mode_flags(overrides: Dict[str, bool]):
    for key, value in overrides.items():
        if key in MODE_FLAGS:
            MODE_FLAGS[key] = bool(value)


def _load_config() -> Dict:
    defaults = {
        "detection_url": os.getenv("UNIFIED_DETECTION_URL", DETECTION_URL),
        "ui_template_path": str(UI_TEMPLATE_PATH),
        "mode_flags": {},
        "debug_presets": {},
    }
    if not CONFIG_PATH.exists() or yaml is None:
        return defaults
    loaded = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        return defaults
    mode_flags = loaded.get("mode_flags", {})
    if not isinstance(mode_flags, dict):
        mode_flags = {}
    return {
        "detection_url": str(loaded.get("detection_url", defaults["detection_url"])),
        "ui_template_path": str(loaded.get("ui_template_path", defaults["ui_template_path"])),
        "mode_flags": mode_flags,
        "debug_presets": loaded.get("debug_presets", {}),
    }


def _model_ui_from_model_path(model_path: str) -> str:
    name = Path(model_path).name.lower()
    if "yolov8n_baseline_multiscale" in name:
        return "yolov8n_baseline_multiscale"
    if "nanodet" in name:
        return "nanodet"
    return "yolov8n_baseline_multiscale"


def _load_debug_preset(preset_name: str) -> Dict:
    presets = dict(DEBUG_PRESET_PATHS)
    cfg = getattr(getattr(globals().get("app", None), "state", None), "config", {}) or {}
    cfg_presets = cfg.get("debug_presets", {})
    if isinstance(cfg_presets, dict):
        for key, value in cfg_presets.items():
            try:
                presets[str(key)] = Path(str(value))
            except Exception:
                continue

    path = presets.get(preset_name)
    if path is None:
        raise HTTPException(status_code=400, detail=f"Неизвестный debug_preset: {preset_name}")
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"Файл debug-профиля не найден: {path}")
    if yaml is None:
        raise HTTPException(status_code=500, detail="PyYAML недоступен")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise HTTPException(status_code=400, detail=f"Некорректный формат debug-профиля: {path}")
    infer = raw.get("infer", {}) if isinstance(raw.get("infer", {}), dict) else {}
    dataset = raw.get("dataset", {}) if isinstance(raw.get("dataset", {}), dict) else {}
    eval_cfg = raw.get("eval", {}) if isinstance(raw.get("eval", {}), dict) else {}
    alert_cfg = raw.get("alert", {}) if isinstance(raw.get("alert", {}), dict) else {}
    runtime_cfg = raw.get("runtime", {}) if isinstance(raw.get("runtime", {}), dict) else {}
    model_path = str(raw.get("model_path", "")).strip()
    thresholds = eval_cfg.get("thresholds", [])
    detector_conf = None
    if isinstance(thresholds, list) and thresholds:
        try:
            detector_conf = float(thresholds[0])
        except Exception:
            detector_conf = None
    if detector_conf is None:
        try:
            detector_conf = float(infer.get("conf_min", 0.25))
        except Exception:
            detector_conf = 0.25
    try:
        target_fps = float(dataset.get("fps", 0.0))
    except Exception:
        target_fps = 0.0
    display_conf = None
    try:
        if "score_threshold" in runtime_cfg:
            display_conf = float(runtime_cfg.get("score_threshold"))
    except Exception:
        display_conf = None
    if display_conf is None:
        try:
            display_conf = float(min([float(x) for x in eval_cfg.get("thresholds", [])])) if isinstance(eval_cfg.get("thresholds", []), list) and eval_cfg.get("thresholds", []) else float(detector_conf)
        except Exception:
            display_conf = float(detector_conf)
    return {
        "name": str(raw.get("name", preset_name)),
        "reason": str(raw.get("reason", "")),
        "model_ui": _model_ui_from_model_path(model_path),
        "detector_conf": max(0.0, min(1.0, float(detector_conf))),
        "infer_conf": max(0.0, min(1.0, float(infer.get("conf_min", detector_conf)))),
        "infer_nms_iou": float(infer.get("nms_iou", 0.7)),
        "infer_max_det": int(infer.get("max_det", 300)),
        "target_fps": target_fps if target_fps > 0 else None,
        "dataset_images_dir": str(dataset.get("images_dir", "")).strip(),
        "dataset_coco_gt": str(dataset.get("coco_gt", "")).strip(),
        "eval_thresholds": [float(x) for x in eval_cfg.get("thresholds", [])] if isinstance(eval_cfg.get("thresholds", []), list) else [],
        "display_conf": max(0.0, min(1.0, float(display_conf))),
        "target_recall": float(eval_cfg["target_recall"]) if isinstance(eval_cfg, dict) and "target_recall" in eval_cfg else None,
        "fp_per_min_target": float(eval_cfg["fp_per_min_target"]) if isinstance(eval_cfg, dict) and "fp_per_min_target" in eval_cfg else None,
        "fp_total_max": float(eval_cfg["fp_total_max"]) if isinstance(eval_cfg, dict) and "fp_total_max" in eval_cfg else 1.0e12,
        "alert_contract": {
            "window_sec": float(alert_cfg.get("window_sec", 1.0)),
            "quorum_k": int(alert_cfg.get("quorum_k", 1)),
            "cooldown_sec": float(alert_cfg.get("cooldown_sec", 1.5)),
            "gap_end_sec": float(alert_cfg.get("gap_end_sec", 1.2)),
            "gt_gap_end_sec": float(alert_cfg.get("gt_gap_end_sec", 1.0)),
            "match_tolerance_sec": float(alert_cfg.get("match_tolerance_sec", 1.2)),
            "min_detections_per_frame": int(alert_cfg.get("min_detections_per_frame", 1)),
        },
        "source_path": str(path),
    }


class OpenCVSource:
    def __init__(self, source: str, mode: str, loop_input: bool = False):
        self.source = source
        self.mode = mode
        self.loop_input = loop_input
        self.cap = cv2.VideoCapture(str(source))

    def is_open(self) -> bool:
        return self.cap.isOpened()

    def fps(self) -> float:
        if not self.cap or not self.cap.isOpened():
            return 30.0
        fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        return fps if fps > 1e-3 else 30.0

    def read(self):
        if not self.cap or not self.cap.isOpened():
            return False, None
        ok, frame = self.cap.read()
        if ok:
            return True, frame

        if self.mode == "rtsp":
            self.cap.release()
            time.sleep(0.2)
            self.cap = cv2.VideoCapture(str(self.source))
            if self.cap.isOpened():
                return self.cap.read()
            return False, None

        if self.loop_input:
            self.cap.release()
            self.cap = cv2.VideoCapture(str(self.source))
            if self.cap.isOpened():
                return self.cap.read()
        return False, None

    def close(self):
        if self.cap is not None:
            self.cap.release()


class FolderSource:
    def __init__(self, folder: str, loop_input: bool = False):
        p = Path(folder)
        if not p.exists() or not p.is_dir():
            raise RuntimeError(f"Папка кадров не найдена: {folder}")
        files = self._scan_image_files(p)
        self.files = sorted(files)
        self.loop_input = loop_input
        self.idx = 0
        self.last_path: Optional[Path] = None

    @staticmethod
    def _scan_image_files(root: Path) -> List[Path]:
        # Поддерживаем рекурсивные структуры миссий с кадрами в подпапках.
        exts = {".jpg", ".jpeg", ".png"}
        files: List[Path] = []
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in exts:
                files.append(path)
        return files

    def is_open(self) -> bool:
        return len(self.files) > 0

    def fps(self) -> float:
        return 30.0

    def read(self):
        if not self.files:
            return False, None
        if self.idx >= len(self.files):
            if not self.loop_input:
                return False, None
            self.idx = 0
        path = self.files[self.idx]
        self.idx += 1
        self.last_path = path
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            return self.read()
        return True, frame

    def close(self):
        return


class BufferedReader:
    def __init__(self, base_reader, buffered_frames: List[np.ndarray]):
        self.base_reader = base_reader
        self.buffer: Deque[np.ndarray] = deque(buffered_frames)

    def is_open(self) -> bool:
        return self.base_reader.is_open()

    def fps(self) -> float:
        return self.base_reader.fps()

    def read(self):
        if self.buffer:
            return True, self.buffer.popleft()
        return self.base_reader.read()

    def close(self):
        self.base_reader.close()


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class YoloDebugEvaluator:
    def __init__(
        self,
        ann_dir: Path,
        iou_thr: float = 0.5,
        thresholds: Optional[List[float]] = None,
        target_recall: Optional[float] = None,
        fp_per_min_target: Optional[float] = None,
        dataset_fps: Optional[float] = None,
        min_detections_per_frame: int = 1,
    ):
        if not ann_dir.exists() or not ann_dir.is_dir():
            raise RuntimeError(f"Папка аннотаций не найдена: {ann_dir}")
        self.ann_dir = ann_dir
        self.iou_thr = float(iou_thr)
        self.index = self._build_index(ann_dir)
        self.gt_total = 0
        self.tp = 0
        self.fp = 0
        self.pred_scores: List[float] = []
        self.pred_tp_flags: List[int] = []
        self.thresholds = [float(x) for x in (thresholds or [])]
        self.target_recall = target_recall
        self.fp_per_min_target = fp_per_min_target
        self.dataset_fps = float(dataset_fps) if dataset_fps and dataset_fps > 0 else None
        self.min_detections_per_frame = max(1, int(min_detections_per_frame))
        self.frame_samples: List[Dict[str, List]] = []

    @staticmethod
    def _build_index(root: Path) -> Dict[str, Path]:
        out: Dict[str, Path] = {}
        for p in root.rglob("*.txt"):
            out[p.stem] = p
        return out

    @staticmethod
    def _load_gt_boxes(path: Optional[Path], img_w: int, img_h: int) -> List[np.ndarray]:
        if path is None or not path.exists():
            return []
        boxes: List[np.ndarray] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            raw = line.strip()
            if not raw:
                continue
            parts = raw.split()
            if len(parts) < 5:
                continue
            try:
                cls_id = int(float(parts[0]))
                # Для debug считаем только person=0.
                if cls_id != 0:
                    continue
                xc, yc, w, h = map(float, parts[1:5])
            except Exception:
                continue
            x1 = (xc - w / 2.0) * img_w
            y1 = (yc - h / 2.0) * img_h
            x2 = (xc + w / 2.0) * img_w
            y2 = (yc + h / 2.0) * img_h
            boxes.append(np.array([x1, y1, x2, y2], dtype=float))
        return boxes

    def update(self, frame_path: Optional[Path], frame_shape: Tuple[int, int], detections: List[Dict]):
        if frame_path is None:
            return
        h, w = frame_shape
        ann_path = self.index.get(frame_path.stem)
        gt_boxes = self._load_gt_boxes(ann_path, w, h)
        self.gt_total += len(gt_boxes)

        preds: List[Tuple[np.ndarray, float]] = []
        for det in detections:
            box = det.get("bbox_xyxy", det.get("box", det.get("bbox", [])))
            score = float(det.get("conf", det.get("score", 0.0)))
            if not box or len(box) < 4:
                continue
            preds.append((np.array(box[:4], dtype=float), score))
        preds.sort(key=lambda x: x[1], reverse=True)

        matched_gt: set[int] = set()
        for p_box, p_score in preds:
            best_iou = 0.0
            best_idx = -1
            for i, gt in enumerate(gt_boxes):
                if i in matched_gt:
                    continue
                iou = _iou_xyxy(p_box, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            is_tp = 1 if (best_idx >= 0 and best_iou >= self.iou_thr) else 0
            self.pred_scores.append(p_score)
            self.pred_tp_flags.append(is_tp)
            if is_tp:
                matched_gt.add(best_idx)
                self.tp += 1
            else:
                self.fp += 1

        self.frame_samples.append({"gt": gt_boxes, "preds": preds})

    @staticmethod
    def _ap_from_scores(scores: List[float], tp_flags: List[int], gt_total: int) -> float:
        if gt_total <= 0 or not scores:
            return 0.0
        order = np.argsort(-np.array(scores))
        tps = np.array(tp_flags, dtype=float)[order]
        fps = 1.0 - tps
        tp_cum = np.cumsum(tps)
        fp_cum = np.cumsum(fps)
        recalls = tp_cum / max(float(gt_total), 1e-9)
        precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
        mrec = np.concatenate(([0.0], recalls, [1.0]))
        mpre = np.concatenate(([0.0], precisions, [0.0]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
        return ap

    def summary(self) -> Dict[str, float]:
        fn = max(0, self.gt_total - self.tp)
        precision = float(self.tp / (self.tp + self.fp)) if (self.tp + self.fp) > 0 else 0.0
        recall = float(self.tp / self.gt_total) if self.gt_total > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        ap50 = self._ap_from_scores(self.pred_scores, self.pred_tp_flags, self.gt_total)
        out = {
            "map50": ap50,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "gt_total": float(self.gt_total),
            "tp": float(self.tp),
            "fp": float(self.fp),
            "fn": float(fn),
        }
        if self.thresholds:
            rows = []
            for thr in self.thresholds:
                m = self._evaluate_threshold(float(thr))
                rows.append({"thr": float(thr), **m})

            feasible = rows
            reason = "min fp by thresholds"
            if self.target_recall is not None:
                feasible = [r for r in feasible if r["recall"] >= float(self.target_recall)]
                reason = f"recall>={float(self.target_recall):.3f}"
            if self.fp_per_min_target is not None:
                feasible = [r for r in feasible if r["fp_per_min"] <= float(self.fp_per_min_target)]
                reason += f" and fp/min<={float(self.fp_per_min_target):.3f}"

            if feasible:
                best = sorted(feasible, key=lambda r: (r["fp_per_min"], r["fp"], -r["precision"], -r["f1"]))[0]
                selection = f"min fp/min under {reason}"
            else:
                best = sorted(rows, key=lambda r: (-r["recall"], r["fp_per_min"], r["fp"], -r["precision"]))[0]
                selection = f"fallback max recall ({reason} not met)"

            out.update(
                {
                    "config_threshold": float(best["thr"]),
                    "config_precision": float(best["precision"]),
                    "config_recall": float(best["recall"]),
                    "config_f1": float(best["f1"]),
                    "config_tp": float(best["tp"]),
                    "config_fp": float(best["fp"]),
                    "config_fn": float(best["fn"]),
                    "config_fp_per_min": float(best["fp_per_min"]),
                    "config_selection": selection,
                    "config_thresholds_count": float(len(self.thresholds)),
                }
            )
        return out

    def _evaluate_threshold(self, thr: float) -> Dict[str, float]:
        tp = 0
        fp = 0
        fn = 0
        for sample in self.frame_samples:
            gts = sample["gt"]
            preds = [p for p in sample["preds"] if float(p[1]) >= thr]
            preds.sort(key=lambda x: float(x[1]), reverse=True)
            if len(preds) < self.min_detections_per_frame:
                preds = []
            matched = [False] * len(gts)
            for p_box, _p_score in preds:
                best = -1
                best_iou = 0.0
                for i, gt in enumerate(gts):
                    if matched[i]:
                        continue
                    iou = _iou_xyxy(p_box, gt)
                    if iou > best_iou:
                        best_iou = iou
                        best = i
                if best >= 0 and best_iou >= self.iou_thr:
                    matched[best] = True
                    tp += 1
                else:
                    fp += 1
            fn += sum(1 for m in matched if not m)

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        if self.dataset_fps and self.dataset_fps > 0 and self.frame_samples:
            duration_min = (len(self.frame_samples) / self.dataset_fps) / 60.0
            fp_per_min = float(fp / duration_min) if duration_min > 0 else float(fp)
        else:
            fp_per_min = float(fp)
        return {
            "tp": float(tp),
            "fp": float(fp),
            "fn": float(fn),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fp_per_min": fp_per_min,
        }


class CocoJsonDebugEvaluator:
    def __init__(
        self,
        coco_gt_path: Path,
        images_root: Optional[Path] = None,
        iou_thr: float = 0.5,
        thresholds: Optional[List[float]] = None,
        target_recall: Optional[float] = None,
        fp_per_min_target: Optional[float] = None,
        dataset_fps: Optional[float] = None,
        min_detections_per_frame: int = 1,
    ):
        if not coco_gt_path.exists() or not coco_gt_path.is_file():
            raise RuntimeError(f"COCO-аннотация не найдена: {coco_gt_path}")
        self.coco_gt_path = coco_gt_path
        self.images_root = images_root
        self.iou_thr = float(iou_thr)
        self.thresholds = [float(x) for x in (thresholds or [])]
        self.target_recall = target_recall
        self.fp_per_min_target = fp_per_min_target
        self.dataset_fps = float(dataset_fps) if dataset_fps and dataset_fps > 0 else None
        self.min_detections_per_frame = max(1, int(min_detections_per_frame))

        self.gt_by_file: Dict[str, List[np.ndarray]] = {}
        self.gt_by_stem: Dict[str, List[np.ndarray]] = {}
        self._load_coco()

        self.gt_total = 0
        self.tp = 0
        self.fp = 0
        self.pred_scores: List[float] = []
        self.pred_tp_flags: List[int] = []
        self.frame_samples: List[Dict[str, List]] = []

    def _load_coco(self):
        data = json.loads(self.coco_gt_path.read_text(encoding="utf-8"))
        images = data.get("images", []) if isinstance(data, dict) else []
        anns = data.get("annotations", []) if isinstance(data, dict) else []
        cats = data.get("categories", []) if isinstance(data, dict) else []

        person_cat_ids = set()
        for c in cats:
            try:
                if str(c.get("name", "")).strip().lower() == "person":
                    person_cat_ids.add(int(c.get("id")))
            except Exception:
                continue
        if not person_cat_ids:
            person_cat_ids = {1}

        id_to_file: Dict[int, str] = {}
        for im in images:
            try:
                iid = int(im.get("id"))
                fname = str(im.get("file_name", "")).replace("\\", "/")
            except Exception:
                continue
            id_to_file[iid] = fname
            self.gt_by_file.setdefault(fname, [])
            self.gt_by_stem.setdefault(Path(fname).stem, [])

        for ann in anns:
            try:
                if int(ann.get("category_id", -1)) not in person_cat_ids:
                    continue
                img_id = int(ann.get("image_id"))
                bbox = ann.get("bbox", [])
                if not isinstance(bbox, list) or len(bbox) < 4:
                    continue
                x, y, w, h = map(float, bbox[:4])
                box = np.array([x, y, x + w, y + h], dtype=float)
            except Exception:
                continue
            fname = id_to_file.get(img_id)
            if not fname:
                continue
            self.gt_by_file.setdefault(fname, []).append(box)
            self.gt_by_stem.setdefault(Path(fname).stem, []).append(box)

    @staticmethod
    def _ap_from_scores(scores: List[float], tp_flags: List[int], gt_total: int) -> float:
        if gt_total <= 0 or not scores:
            return 0.0
        order = np.argsort(-np.array(scores))
        tps = np.array(tp_flags, dtype=float)[order]
        fps = 1.0 - tps
        tp_cum = np.cumsum(tps)
        fp_cum = np.cumsum(fps)
        recalls = tp_cum / max(float(gt_total), 1e-9)
        precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
        mrec = np.concatenate(([0.0], recalls, [1.0]))
        mpre = np.concatenate(([0.0], precisions, [0.0]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
        return ap

    def _resolve_gt_boxes(self, frame_path: Optional[Path]) -> List[np.ndarray]:
        if frame_path is None:
            return []
        key = None
        fp = Path(frame_path)
        if self.images_root is not None:
            try:
                rel = fp.resolve().relative_to(self.images_root.resolve())
                key = str(rel).replace("\\", "/")
            except Exception:
                key = None
        if key and key in self.gt_by_file:
            return self.gt_by_file.get(key, [])
        by_name = self.gt_by_file.get(fp.name.replace("\\", "/"))
        if by_name is not None:
            return by_name
        return self.gt_by_stem.get(fp.stem, [])

    def update(self, frame_path: Optional[Path], frame_shape: Tuple[int, int], detections: List[Dict]):
        _ = frame_shape
        gt_boxes = self._resolve_gt_boxes(frame_path)
        self.gt_total += len(gt_boxes)

        preds: List[Tuple[np.ndarray, float]] = []
        for det in detections:
            box = det.get("bbox_xyxy", det.get("box", det.get("bbox", [])))
            score = float(det.get("conf", det.get("score", 0.0)))
            if not box or len(box) < 4:
                continue
            preds.append((np.array(box[:4], dtype=float), score))
        preds.sort(key=lambda x: x[1], reverse=True)

        matched_gt: set[int] = set()
        for p_box, p_score in preds:
            best_iou = 0.0
            best_idx = -1
            for i, gt in enumerate(gt_boxes):
                if i in matched_gt:
                    continue
                iou = _iou_xyxy(p_box, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            is_tp = 1 if (best_idx >= 0 and best_iou >= self.iou_thr) else 0
            self.pred_scores.append(p_score)
            self.pred_tp_flags.append(is_tp)
            if is_tp:
                matched_gt.add(best_idx)
                self.tp += 1
            else:
                self.fp += 1

        self.frame_samples.append({"gt": gt_boxes, "preds": preds})

    def _evaluate_threshold(self, thr: float) -> Dict[str, float]:
        tp = 0
        fp = 0
        fn = 0
        for sample in self.frame_samples:
            gts = sample["gt"]
            preds = [p for p in sample["preds"] if float(p[1]) >= thr]
            preds.sort(key=lambda x: float(x[1]), reverse=True)
            if len(preds) < self.min_detections_per_frame:
                preds = []
            matched = [False] * len(gts)
            for p_box, _p_score in preds:
                best = -1
                best_iou = 0.0
                for i, gt in enumerate(gts):
                    if matched[i]:
                        continue
                    iou = _iou_xyxy(p_box, gt)
                    if iou > best_iou:
                        best_iou = iou
                        best = i
                if best >= 0 and best_iou >= self.iou_thr:
                    matched[best] = True
                    tp += 1
                else:
                    fp += 1
            fn += sum(1 for m in matched if not m)

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        if self.dataset_fps and self.dataset_fps > 0 and self.frame_samples:
            duration_min = (len(self.frame_samples) / self.dataset_fps) / 60.0
            fp_per_min = float(fp / duration_min) if duration_min > 0 else float(fp)
        else:
            fp_per_min = float(fp)
        return {
            "tp": float(tp),
            "fp": float(fp),
            "fn": float(fn),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fp_per_min": fp_per_min,
        }

    def summary(self) -> Dict[str, float]:
        fn = max(0, self.gt_total - self.tp)
        precision = float(self.tp / (self.tp + self.fp)) if (self.tp + self.fp) > 0 else 0.0
        recall = float(self.tp / self.gt_total) if self.gt_total > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        ap50 = self._ap_from_scores(self.pred_scores, self.pred_tp_flags, self.gt_total)
        out = {
            "map50": ap50,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "gt_total": float(self.gt_total),
            "tp": float(self.tp),
            "fp": float(self.fp),
            "fn": float(fn),
        }
        if self.thresholds:
            rows = []
            for thr in self.thresholds:
                m = self._evaluate_threshold(float(thr))
                rows.append({"thr": float(thr), **m})

            feasible = rows
            reason = "min fp by thresholds"
            if self.target_recall is not None:
                feasible = [r for r in feasible if r["recall"] >= float(self.target_recall)]
                reason = f"recall>={float(self.target_recall):.3f}"
            if self.fp_per_min_target is not None:
                feasible = [r for r in feasible if r["fp_per_min"] <= float(self.fp_per_min_target)]
                reason += f" and fp/min<={float(self.fp_per_min_target):.3f}"

            if feasible:
                best = sorted(feasible, key=lambda r: (r["fp_per_min"], r["fp"], -r["precision"], -r["f1"]))[0]
                selection = f"min fp/min under {reason}"
            else:
                best = sorted(rows, key=lambda r: (-r["recall"], r["fp_per_min"], r["fp"], -r["precision"]))[0]
                selection = f"fallback max recall ({reason} not met)"

            out.update(
                {
                    "config_threshold": float(best["thr"]),
                    "config_precision": float(best["precision"]),
                    "config_recall": float(best["recall"]),
                    "config_f1": float(best["f1"]),
                    "config_tp": float(best["tp"]),
                    "config_fp": float(best["fp"]),
                    "config_fn": float(best["fn"]),
                    "config_fp_per_min": float(best["fp_per_min"]),
                    "config_selection": selection,
                    "config_thresholds_count": float(len(self.thresholds)),
                }
            )
        return out


def _mission_and_frame(file_name: str) -> Tuple[str, int]:
    stem = Path(file_name).stem
    parts = stem.split("_")
    if len(parts) >= 3 and parts[2].isdigit():
        return "_".join(parts[:2]), int(parts[2])
    frame_digits = "".join(ch for ch in stem if ch.isdigit())
    frame_num = int(frame_digits) if frame_digits else 0
    return parts[0] if parts else stem, frame_num


class CocoAlertContractEvaluator:
    def __init__(self, coco_gt_path: Path, images_root: Optional[Path], cfg: Dict[str, Any]):
        if not coco_gt_path.exists() or not coco_gt_path.is_file():
            raise RuntimeError(f"COCO-аннотация не найдена: {coco_gt_path}")
        self.coco_gt_path = coco_gt_path
        self.images_root = images_root
        self.thresholds = [float(x) for x in cfg.get("thresholds", [])]
        self.target_recall = cfg.get("target_recall")
        self.fp_per_min_target = cfg.get("fp_per_min_target")
        self.fp_total_max = float(cfg.get("fp_total_max", 1.0e12))
        self.fps = float(cfg.get("fps", 6.0))
        self.alert = dict(cfg.get("alert_contract", {}))
        self.alert["min_detections_per_frame"] = max(1, int(self.alert.get("min_detections_per_frame", 1)))

        self._frame_meta = self._load_frame_meta()
        self._obs_by_mission: Dict[str, List[Dict[str, Any]]] = {}

    def _load_frame_meta(self) -> Dict[str, Dict[str, Any]]:
        payload = json.loads(self.coco_gt_path.read_text(encoding="utf-8"))
        images = payload.get("images", []) if isinstance(payload, dict) else []
        anns = payload.get("annotations", []) if isinstance(payload, dict) else []
        cats = payload.get("categories", []) if isinstance(payload, dict) else []

        person_cat_ids = set()
        for c in cats:
            try:
                if str(c.get("name", "")).strip().lower() == "person":
                    person_cat_ids.add(int(c.get("id")))
            except Exception:
                continue
        if not person_cat_ids:
            person_cat_ids = {1}

        gt_by_image: Dict[int, bool] = {}
        for ann in anns:
            try:
                if int(ann.get("category_id", -1)) in person_cat_ids:
                    gt_by_image[int(ann.get("image_id"))] = True
            except Exception:
                continue

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for im in images:
            try:
                iid = int(im.get("id"))
                file_name = str(im.get("file_name", "")).replace("\\", "/")
            except Exception:
                continue
            mission_id, frame_num = _mission_and_frame(file_name)
            grouped.setdefault(mission_id, []).append(
                {
                    "image_id": iid,
                    "file_name": file_name,
                    "frame_num": int(frame_num),
                    "gt_present": bool(gt_by_image.get(iid, False)),
                }
            )

        out: Dict[str, Dict[str, Any]] = {}
        for mission_id, items in grouped.items():
            items.sort(key=lambda x: int(x["frame_num"]))
            first_frame_num = int(items[0]["frame_num"]) if items else 0
            for it in items:
                ts_sec = (int(it["frame_num"]) - first_frame_num) / self.fps if self.fps > 0 else 0.0
                meta = {
                    "mission_id": mission_id,
                    "file_name": it["file_name"],
                    "frame_num": int(it["frame_num"]),
                    "ts_sec": float(ts_sec),
                    "gt_present": bool(it["gt_present"]),
                }
                rel = it["file_name"]
                out[rel] = meta
                out[Path(rel).name] = meta
                out[Path(rel).stem] = meta
        return out

    def _resolve_meta(self, frame_path: Optional[Path]) -> Optional[Dict[str, Any]]:
        if frame_path is None:
            return None
        fp = Path(frame_path)
        key = None
        if self.images_root is not None:
            try:
                rel = fp.resolve().relative_to(self.images_root.resolve())
                key = str(rel).replace("\\", "/")
            except Exception:
                key = None
        if key and key in self._frame_meta:
            return self._frame_meta[key]
        if fp.name in self._frame_meta:
            return self._frame_meta[fp.name]
        return self._frame_meta.get(fp.stem)

    def update(self, frame_path: Optional[Path], frame_shape: Tuple[int, int], detections: List[Dict]):
        _ = frame_shape
        meta = self._resolve_meta(frame_path)
        if meta is None:
            return
        scores: List[float] = []
        for det in detections:
            try:
                scores.append(float(det.get("conf", det.get("score", 0.0))))
            except Exception:
                continue
        rec = {
            "ts_sec": float(meta["ts_sec"]),
            "frame_num": int(meta["frame_num"]),
            "gt_present": bool(meta["gt_present"]),
            "scores": np.array(scores, dtype=float),
        }
        self._obs_by_mission.setdefault(str(meta["mission_id"]), []).append(rec)

    @staticmethod
    def _build_gt_episodes(frames: List[Dict[str, Any]], gt_gap_end_sec: float) -> List[Tuple[float, float]]:
        episodes: List[Tuple[float, float]] = []
        start: Optional[float] = None
        end: Optional[float] = None
        for frame in frames:
            ts = float(frame["ts_sec"])
            if bool(frame["gt_present"]):
                if start is None:
                    start = ts
                    end = ts
                else:
                    assert end is not None
                    if ts - end > gt_gap_end_sec:
                        episodes.append((start, end))
                        start = ts
                    end = ts
                continue
            if start is not None and end is not None and ts - end > gt_gap_end_sec:
                episodes.append((start, end))
                start = None
                end = None
        if start is not None and end is not None:
            episodes.append((start, end))
        return episodes

    @staticmethod
    def _build_alerts(frames: List[Dict[str, Any]], threshold: float, alert_cfg: Dict[str, Any]) -> List[float]:
        window_sec = float(alert_cfg.get("window_sec", 1.0))
        quorum_k = int(alert_cfg.get("quorum_k", 1))
        cooldown_sec = float(alert_cfg.get("cooldown_sec", 1.5))
        gap_end_sec = float(alert_cfg.get("gap_end_sec", 1.2))
        min_dets = int(alert_cfg.get("min_detections_per_frame", 1))

        recent_hits: List[float] = []
        last_alert_ts: Optional[float] = None
        last_positive_ts: Optional[float] = None
        alerts: List[float] = []

        for frame in frames:
            ts = float(frame["ts_sec"])
            scores: np.ndarray = frame["scores"]
            positive_count = int((scores >= threshold).sum())
            is_positive = positive_count >= min_dets
            if not is_positive:
                if last_positive_ts is not None and ts - last_positive_ts > gap_end_sec:
                    recent_hits.clear()
                continue

            if last_positive_ts is not None and ts - last_positive_ts > gap_end_sec:
                recent_hits.clear()
            recent_hits.append(ts)
            last_positive_ts = ts
            lower = ts - window_sec
            recent_hits = [x for x in recent_hits if x >= lower]
            if len(recent_hits) < quorum_k:
                continue
            if last_alert_ts is not None and ts - last_alert_ts < cooldown_sec:
                continue
            last_alert_ts = ts
            alerts.append(ts)
        return alerts

    @staticmethod
    def _episode_found(episode: Tuple[float, float], alert_ts: List[float], tau: float) -> bool:
        lo = float(episode[0]) - tau
        hi = float(episode[1]) + tau
        return any(lo <= ts <= hi for ts in alert_ts)

    def _evaluate_threshold(self, threshold: float) -> Dict[str, float]:
        tau = float(self.alert.get("match_tolerance_sec", 1.2))
        gt_gap = float(self.alert.get("gt_gap_end_sec", 1.0))
        total_episodes = 0
        total_found = 0
        total_alerts = 0
        total_false = 0
        total_duration = 0.0
        total_duration_count = 0.0

        for mission_id, frames in self._obs_by_mission.items():
            _ = mission_id
            frames_sorted = sorted(frames, key=lambda x: int(x["frame_num"]))
            if not frames_sorted:
                continue
            episodes = self._build_gt_episodes(frames_sorted, gt_gap_end_sec=gt_gap)
            alerts = self._build_alerts(frames_sorted, threshold=threshold, alert_cfg=self.alert)
            found = sum(1 for ep in episodes if self._episode_found(ep, alerts, tau))
            fp = 0
            for ats in alerts:
                if not any((ep[0] - tau) <= ats <= (ep[1] + tau) for ep in episodes):
                    fp += 1
            dur_sec = float(frames_sorted[-1]["ts_sec"] - frames_sorted[0]["ts_sec"]) if len(frames_sorted) > 1 else 0.0
            dur_count_sec = (max(0, len(frames_sorted) - 1) / self.fps) if self.fps > 0 else 0.0
            total_episodes += len(episodes)
            total_found += found
            total_alerts += len(alerts)
            total_false += fp
            total_duration += max(0.0, dur_sec)
            total_duration_count += max(0.0, dur_count_sec)

        duration_min = total_duration / 60.0 if total_duration > 0 else 0.0
        fp_per_min = (float(total_false) / duration_min) if duration_min > 0 else 0.0
        recall_event = (float(total_found) / float(total_episodes)) if total_episodes > 0 else 0.0
        return {
            "threshold": float(threshold),
            "episodes_total": float(total_episodes),
            "episodes_found": float(total_found),
            "recall_event": float(recall_event),
            "alerts_total": float(total_alerts),
            "false_alerts_total": float(total_false),
            "fp_per_min": float(fp_per_min),
            "duration_sec": float(total_duration),
            "duration_count_sec": float(total_duration_count),
        }

    def summary(self) -> Dict[str, float]:
        if not self.thresholds:
            return {"recall_event": 0.0, "fp_per_min": 0.0, "episodes_total": 0.0, "episodes_found": 0.0}
        results = [self._evaluate_threshold(thr) for thr in self.thresholds]
        fp_target = float(self.fp_per_min_target) if self.fp_per_min_target is not None else 1.0e12
        feasible = [
            r
            for r in results
            if r["fp_per_min"] <= fp_target and r["false_alerts_total"] <= self.fp_total_max
        ]
        if feasible:
            best = sorted(
                feasible,
                key=lambda r: (r["recall_event"], -r["fp_per_min"]),
                reverse=True,
            )[0]
            selection = "constraints_met"
        else:
            soft = [r for r in results if r["false_alerts_total"] <= self.fp_total_max]
            if soft:
                best = sorted(
                    soft,
                    key=lambda r: (r["recall_event"], -r["fp_per_min"]),
                    reverse=True,
                )[0]
                selection = "best_under_fp_total"
            else:
                best = sorted(
                    results,
                    key=lambda r: (r["recall_event"], -r["fp_per_min"]),
                    reverse=True,
                )[0]
                selection = "best_recall_only"
        return {
            "contract_mode": "alert_episode",
            "recall_event": float(best["recall_event"]),
            "fp_per_min": float(best["fp_per_min"]),
            "episodes_total": float(best["episodes_total"]),
            "episodes_found": float(best["episodes_found"]),
            "alerts_total": float(best["alerts_total"]),
            "false_alerts_total": float(best["false_alerts_total"]),
            "threshold": float(best["threshold"]),
            "duration_sec": float(best.get("duration_sec", 0.0)),
            "duration_count_sec": float(best.get("duration_count_sec", 0.0)),
            "selection": selection,
        }


def detect_red_square_corners(frame_bgr):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 80, 80), (10, 255, 255)) | cv2.inRange(hsv, (170, 80, 80), (180, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 1500:
        return None
    approx = cv2.approxPolyDP(cnt, 0.03 * cv2.arcLength(cnt, True), True)
    if len(approx) != 4:
        return None
    pts = approx.reshape(4, 2).astype(np.float32)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    return pts[np.argsort(ang)]


def detect_red_square_corners_alt(frame_bgr):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 80, 80), (10, 255, 255)) | cv2.inRange(hsv, (170, 80, 80), (180, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), dtype=np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), dtype=np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 1500:
        return None
    approx = cv2.approxPolyDP(cnt, 0.03 * cv2.arcLength(cnt, True), True)
    if len(approx) != 4:
        return None
    pts = approx.reshape(4, 2).astype(np.float32)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    return pts[np.argsort(ang)]


def estimate_camera_center_from_marker(corners_px, K, marker_size_m: float = 1.0):
    marker_3d = MARKER_3D * float(marker_size_m)
    ok, _, tvec = cv2.solvePnP(marker_3d, corners_px, K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    t = tvec.flatten()
    return np.array([t[0], t[1], abs(t[2])], dtype=float)


def compute_scale_from_samples(samples):
    if samples is None or len(samples) < 2:
        return None
    c = np.median(samples, axis=0)
    d = np.linalg.norm(samples - c, axis=1)
    return np.median(d) if len(d) else None


def order_points(pts4):
    pts = np.array(pts4, dtype=np.float32)
    s = pts.sum(axis=1)
    d = pts[:, 0] - pts[:, 1]
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmax(d)]
    bl = pts[np.argmin(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def polygon_area(pts4):
    return float(abs(cv2.contourArea(np.array(pts4, dtype=np.float32).reshape(-1, 1, 2))))


def make_roi_mask(gray, roi_top_ratio=0.45):
    h, w = gray.shape
    m = np.zeros((h, w), dtype=np.uint8)
    m[int(h * roi_top_ratio):, :] = 255
    return m


def safe_inv_homography(H):
    if H is None:
        return None
    H = H.astype(np.float64)
    if not np.isfinite(H).all():
        return None

    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]

    try:
        cond = np.linalg.cond(H)
    except Exception:
        return None
    if not np.isfinite(cond) or cond > 1e7:
        return None

    try:
        invH = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None

    if not np.isfinite(invH).all():
        return None
    if abs(invH[2, 2]) > 1e-12:
        invH = invH / invH[2, 2]
    return invH


def project_point(H_img_to_plane, x, y):
    pt = np.array([[[float(x), float(y)]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, H_img_to_plane.astype(np.float32))[0, 0]
    if not np.isfinite(out).all():
        return None
    return float(out[0]), float(out[1])


def project_points_median(H_img_to_plane, pts_xy):
    if pts_xy is None or len(pts_xy) < 10:
        return None

    pts = pts_xy.reshape(-1, 1, 2).astype(np.float32)
    plane = cv2.perspectiveTransform(pts, H_img_to_plane.astype(np.float32)).reshape(-1, 2)

    ok = np.isfinite(plane).all(axis=1)
    plane = plane[ok]
    if len(plane) < 10:
        return None

    mx = float(np.median(plane[:, 0]))
    my = float(np.median(plane[:, 1]))
    return mx, my


def preprocess_gray(frame_bgr, use_clahe=False, clahe_clip=2.0, clahe_grid=8):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    if use_clahe:
        g = max(2, int(clahe_grid))
        clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=(g, g))
        gray = clahe.apply(gray)
    return gray


def detect_red_marker_corners(frame_bgr):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    lower1 = np.array([0, 120, 60], dtype=np.uint8)
    upper1 = np.array([10, 255, 255], dtype=np.uint8)
    lower2 = np.array([170, 120, 60], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)

    k = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return None, mask

    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    area = int(stats[largest, cv2.CC_STAT_AREA])
    if area < 2500:
        return None, mask

    mask = (labels == largest).astype(np.uint8) * 255

    edges = cv2.Canny(mask, 50, 150)
    corners = cv2.goodFeaturesToTrack(
        edges, maxCorners=160, qualityLevel=0.01, minDistance=18, blockSize=7
    )

    if corners is None or len(corners) < 4:
        ys, xs = np.where(mask == 255)
        if len(xs) < 2000:
            return None, mask
        pts = np.stack([xs, ys], axis=1).astype(np.float32)
    else:
        pts = corners.reshape(-1, 2).astype(np.float32)

    pts4 = order_points(
        [
            pts[np.argmin(pts[:, 0] + pts[:, 1])],
            pts[np.argmax(pts[:, 0] - pts[:, 1])],
            pts[np.argmax(pts[:, 0] + pts[:, 1])],
            pts[np.argmin(pts[:, 0] - pts[:, 1])],
        ]
    )

    dil = cv2.dilate(mask, np.ones((9, 9), np.uint8), iterations=1)
    h, w = mask.shape
    for p in pts4:
        x, y = int(round(p[0])), int(round(p[1]))
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))
        if dil[y, x] == 0:
            return None, mask

    return pts4, mask


def _laplacian_smooth_last(traj_points: List[np.ndarray], pos: np.ndarray) -> np.ndarray:
    if len(traj_points) < 2:
        return pos
    window = traj_points[-SMOOTH_WINDOW:] + [pos]
    n = len(window)
    if n < 3:
        return pos
    A = np.zeros((n, n), dtype=float)
    for i in range(n - 1):
        A[i, i + 1] = 1.0
        A[i + 1, i] = 1.0
    D = np.diag(A.sum(axis=1))
    L = D - A
    X = np.vstack(window)
    grad = 2.0 * (L @ X)[-1]
    pos_smoothed = pos.copy()
    pos_smoothed[:2] -= SMOOTH_LR_XY * grad[:2]
    pos_smoothed[2] -= SMOOTH_LR_Z * grad[2]
    return pos_smoothed


def draw_detections(frame_bgr: np.ndarray, detections: List[Dict]):
    for det in detections:
        box = det.get("bbox_xyxy", det.get("box", det.get("bbox", [])))
        score = float(det.get("conf", det.get("score", 0.0)))
        if not box or len(box) < 4:
            continue
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(
            frame_bgr,
            f"Person {score:.2f}",
            (x1, max(y1 - 6, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 200, 0),
            2,
            cv2.LINE_AA,
        )


def _scale_detections_xyxy(detections: List[Dict], src_shape: Tuple[int, int], dst_shape: Tuple[int, int]) -> List[Dict]:
    if not detections:
        return detections
    src_h, src_w = src_shape
    dst_h, dst_w = dst_shape
    if src_h <= 0 or src_w <= 0 or (src_h == dst_h and src_w == dst_w):
        return detections
    sx = float(dst_w) / float(src_w)
    sy = float(dst_h) / float(src_h)
    out: List[Dict] = []
    for d in detections:
        box = d.get("bbox_xyxy", d.get("box", d.get("bbox", [])))
        if not box or len(box) < 4:
            out.append(d)
            continue
        x1, y1, x2, y2 = map(float, box[:4])
        db = dict(d)
        db["bbox_xyxy"] = [x1 * sx, y1 * sy, x2 * sx, y2 * sy]
        out.append(db)
    return out


def encode_image_b64(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("utf-8")


def render_plots(traj_world_np: np.ndarray, times: List[float]):
    xs, ys, zs = traj_world_np[:, 0], traj_world_np[:, 1], traj_world_np[:, 2]
    t_np = np.array(times)

    def padded(vals):
        vmin = float(np.min(vals))
        vmax = float(np.max(vals))
        span = vmax - vmin
        pad = max(1.0, span * 0.1)
        return vmin - pad, vmax + pad

    plots: Dict[str, str] = {}

    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(xs, ys, zs, "-b")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_xlim(*padded(xs))
    ax.set_ylim(*padded(ys))
    ax.set_zlim(*padded(zs))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plots["trajectory3d"] = "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(xs, ys, "-g")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal", "box")
    ax.set_xlim(*padded(xs))
    ax.set_ylim(*padded(ys))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plots["topdown"] = "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(t_np, zs, "-r")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("Z (m)")
    ax.set_xlim(t_np.min(), t_np.max() if t_np.max() > t_np.min() else t_np.min() + 1e-3)
    ax.set_ylim(*padded(zs))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plots["height"] = "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return plots


def save_report(
    report_stem: str,
    traj_world_np: np.ndarray,
    times: List[float],
    person_counts: List[int],
    fps_samples: List[float],
    det_latency_samples: List[float],
    mode_label: str,
    video_path: Optional[Path] = None,
    alert_frames_dir: Optional[Path] = None,
):
    out_dir = REPORT_DIR / report_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "trajectory.csv"
    rows = ["frame,timestamp,x_m,y_m,z_m,person_count,fps,detector_latency_ms,mode\n"]
    for i, (t, p) in enumerate(zip(times, traj_world_np)):
        pc = person_counts[i] if i < len(person_counts) else 0
        fps_v = fps_samples[i] if i < len(fps_samples) else 0.0
        lat_v = det_latency_samples[i] if i < len(det_latency_samples) else 0.0
        rows.append(f"{i},{t:.6f},{p[0]:.6f},{p[1]:.6f},{p[2]:.6f},{pc},{fps_v:.3f},{lat_v:.3f},{mode_label}\n")
    csv_path.write_text("".join(rows), encoding="utf-8")

    xs, ys, zs = traj_world_np[:, 0], traj_world_np[:, 1], traj_world_np[:, 2]
    t_np = np.array(times)

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(xs, ys, zs, "-b", linewidth=3)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    fig.savefig(out_dir / "trajectory_3d.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(xs, ys, "-g", linewidth=3)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal", "box")
    fig.savefig(out_dir / "topdown_xy.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t_np, zs, "-r", linewidth=3)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("Z (m)")
    fig.savefig(out_dir / "height_over_time.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    zip_path = REPORT_DIR / f"{report_stem}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, csv_path.name)
        zf.write(out_dir / "trajectory_3d.png", "trajectory_3d.png")
        zf.write(out_dir / "topdown_xy.png", "topdown_xy.png")
        zf.write(out_dir / "height_over_time.png", "height_over_time.png")
        if video_path and video_path.exists():
            zf.write(video_path, video_path.name)
        if alert_frames_dir and alert_frames_dir.exists():
            for p in sorted(alert_frames_dir.rglob("*")):
                if not p.is_file():
                    continue
                rel = p.relative_to(alert_frames_dir)
                zf.write(p, f"alerts/{rel}")
    return zip_path


def start_remote_source(
    rpi_url: str,
    source_kind: str,
    source_value: str,
    loop_input: bool = False,
    mission_id: str = "",
    realtime: bool = True,
    target_fps: float | None = None,
    jitter_ms: int = 0,
    drop_if_lag: bool = True,
    max_duration_sec: float | None = None,
    jpeg_quality: int = 80,
) -> Tuple[str, str, str, Dict[str, Any]]:
    if not rpi_url:
        raise RuntimeError("Для потокового режима НСУ требуется URL RaspberryPi")
    mode_map = {"video": "file", "rtsp": "rtsp", "frames": "frames"}
    remote_mode = mode_map.get(source_kind)
    if remote_mode is None:
        raise RuntimeError(f"Неизвестный тип источника: {source_kind}")

    payload = {
        "mode": remote_mode,
        "source": source_value,
        "loop": bool(loop_input),
        "jpeg_quality": max(10, min(100, int(jpeg_quality))),
        "mission_id": mission_id,
        "realtime": bool(realtime),
        "target_fps": float(target_fps) if target_fps is not None else 0.0,
        "jitter_ms": max(0, int(jitter_ms)),
        "drop_if_lag": bool(drop_if_lag),
        "max_duration_sec": float(max_duration_sec) if max_duration_sec is not None else 0.0,
    }
    base = rpi_url.strip().rstrip("/")
    resp = requests.post(f"{base}/source/start", json=payload, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    remote_session = data.get("session_id")
    stream_url = data.get("stream_url")
    if not remote_session or not stream_url:
        raise RuntimeError("RPi source service вернул неполный ответ")

    if stream_url.startswith("http://") or stream_url.startswith("https://"):
        absolute_stream = stream_url
    else:
        absolute_stream = f"{base}{stream_url}"
    return remote_session, absolute_stream, base, data


def stop_remote_source(rpi_base_url: str, remote_session_id: str):
    try:
        requests.post(f"{rpi_base_url}/source/stop/{remote_session_id}", timeout=5)
    except Exception:
        pass


def fetch_remote_source_stats(rpi_base_url: str, remote_session_id: str) -> Optional[Dict[str, Any]]:
    try:
        resp = requests.get(f"{rpi_base_url}/source/session/{remote_session_id}", timeout=2)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None


def _validate_mode_chain(run_mode: str, nsu_channel: str, source_kind: str):
    if run_mode == "edge":
        if not is_enabled("enable_edge"):
            raise HTTPException(status_code=400, detail="Edge режим отключен конфигурацией")
        return

    if not is_enabled("enable_nsu"):
        raise HTTPException(status_code=400, detail="НСУ режим отключен конфигурацией")

    if nsu_channel == "local":
        if not is_enabled("enable_nsu_local"):
            raise HTTPException(status_code=400, detail="НСУ локальный режим отключен")
        key = {
            "video": "enable_nsu_local_video",
            "rtsp": "enable_nsu_local_rtsp",
            "frames": "enable_nsu_local_frames",
        }.get(source_kind)
    else:
        if not is_enabled("enable_nsu_stream"):
            raise HTTPException(status_code=400, detail="НСУ потоковый режим отключен")
        key = {
            "video": "enable_nsu_stream_video",
            "rtsp": "enable_nsu_stream_rtsp",
            "frames": "enable_nsu_stream_frames",
        }.get(source_kind)

    if key is None or not is_enabled(key):
        raise HTTPException(status_code=400, detail="Выбранный режим источника отключен")


def _resolve_detector(model_ui: str) -> str:
    mapped = ALLOWED_NSU_MODELS.get(model_ui)
    if not mapped:
        allowed = ", ".join(sorted(ALLOWED_NSU_MODELS.keys()))
        raise HTTPException(status_code=400, detail=f"Недопустимая модель для НСУ режима. Разрешено: {allowed}")
    return mapped


def _autodetect_annotations_dir(frames_root: Path) -> str:
    candidates = [
        frames_root / "labels",
        frames_root / "annotations",
        frames_root.parent / "labels",
        frames_root.parent / "annotations",
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            has_txt = any(c.rglob("*.txt"))
            if has_txt:
                return str(c)
    return ""


def _ensure_rtsp_url(url: str):
    val = (url or "").strip()
    if not val:
        raise HTTPException(status_code=400, detail="RTSP URL пустой")
    lowered = val.lower()
    if not (lowered.startswith("rtsp://") or lowered.startswith("rtsps://")):
        raise HTTPException(status_code=400, detail="Разрешены только RTSP URL (rtsp:// или rtsps://)")


def _probe_marker(reader, seconds: float) -> Tuple[bool, List[np.ndarray]]:
    fps = max(reader.fps(), 1.0)
    limit = max(3, int(seconds * fps))
    buffered: List[np.ndarray] = []
    found = False
    for _ in range(limit):
        ok, frame = reader.read()
        if not ok or frame is None:
            break
        buffered.append(frame)
        frame_marker = cv2.resize(frame, (MARKER_RESIZE_W, MARKER_RESIZE_H))
        marker_pts, _ = detect_red_marker_corners(frame_marker)
        if marker_pts is not None:
            found = True
            break
    return found, buffered


def run_unified_pipeline(
    session_id: str,
    reader,
    detection_client: Optional[DetectionClient],
    on_update: Callable[[Dict], None],
    profile: SourceProfile,
    marker_mode: str,
    save_video: bool,
    report_stem: str,
    detector_name: str,
    detector_conf: float | None,
    detector_iou: float | None,
    detector_max_det: int | None,
    display_conf: float | None,
    target_fps: float | None,
    mode_label: str,
    debug_evaluator: Optional[YoloDebugEvaluator] = None,
    debug_metrics_info: str = "",
    remote_stats_getter: Optional[Callable[[], Optional[Dict[str, Any]]]] = None,
):
    fps_samples: List[float] = []
    person_samples: List[int] = []
    det_latency_samples: List[float] = []
    traj_points: List[np.ndarray] = []
    time_stamps: List[float] = []
    t_start = time.perf_counter()

    if not reader.is_open():
        raise RuntimeError("Не удалось открыть источник")

    has_marker = False
    buffered: List[np.ndarray] = []
    nav_mode = marker_mode
    if marker_mode == "auto":
        has_marker, buffered = _probe_marker(reader, AUTO_MARKER_SECONDS)
        nav_mode = "marker" if has_marker else "no_marker"

    run_reader = BufferedReader(reader, buffered)
    ok, first = run_reader.read()
    if not ok or first is None:
        raise RuntimeError("Не удалось прочитать первый кадр")

    first_orig = first
    first_small = cv2.resize(first_orig, (NAV_W, NAV_H))
    first_marker = cv2.resize(first_orig, (MARKER_RESIZE_W, MARKER_RESIZE_H))
    pending_frames: Deque[np.ndarray] = deque()
    fps = run_reader.fps()
    source_frame_idx = 0

    force_fixed_height_frames = mode_label.endswith(":frames")
    fixed_height_m = 300.0
    last_remote_stats_poll = 0.0
    remote_stats: Optional[Dict[str, Any]] = None

    if nav_mode == "marker":
        pos = np.array([0.0, 0.0, 1.5], dtype=float)
    else:
        pos = np.array([0.0, 0.0, 0.0], dtype=float)
    if force_fixed_height_frames:
        pos[2] = fixed_height_m

    traj_points.append(pos.copy())
    time_stamps.append(0.0)

    frame_idx = 1
    writer = None
    video_out_path = None
    alert_frames_dir: Optional[Path] = None
    alert_saved_count = 0

    if nav_mode == "marker":
        max_init_frames = max(1, int(AUTO_MARKER_SECONDS * fps))
        init_frames: List[np.ndarray] = [first_orig]
        candidates: List[Tuple[int, float, np.ndarray]] = []
        pts0, _ = detect_red_marker_corners(first_marker)
        if pts0 is not None:
            a0 = polygon_area(pts0)
            if 3000 < a0 < (MARKER_RESIZE_W * MARKER_RESIZE_H) * 0.5:
                candidates.append((0, a0, pts0))

        for fi in range(1, max_init_frames):
            ok_i, frame_i = run_reader.read()
            if not ok_i or frame_i is None:
                break
            init_frames.append(frame_i)
            frame_marker_i = cv2.resize(frame_i, (MARKER_RESIZE_W, MARKER_RESIZE_H))
            pts_i, _ = detect_red_marker_corners(frame_marker_i)
            if pts_i is not None:
                ai = polygon_area(pts_i)
                if 3000 < ai < (MARKER_RESIZE_W * MARKER_RESIZE_H) * 0.5:
                    candidates.append((fi, ai, pts_i))

        if not candidates:
            raise RuntimeError("Marker not found in init window.")

        candidates.sort(key=lambda x: x[1], reverse=True)
        topK = candidates[: min(40, len(candidates))]
        stack = np.stack([c[2] for c in topK], axis=0)
        median_corners = np.median(stack, axis=0)

        def dist_to_median(c):
            return float(np.linalg.norm(c[2] - median_corners))

        best = min(topK, key=dist_to_median)
        start_frame_idx = int(best[0])
        best_pts4 = best[2]
        source_frame_idx = start_frame_idx
        first_orig = init_frames[start_frame_idx]
        first_small = cv2.resize(first_orig, (NAV_W, NAV_H))
        first_marker = cv2.resize(first_orig, (MARKER_RESIZE_W, MARKER_RESIZE_H))
        pending_frames = deque(init_frames[start_frame_idx + 1 :])
        H_init = cv2.getPerspectiveTransform(
            best_pts4.astype(np.float32),
            np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]], dtype=np.float32),
        ).astype(np.float64)
    else:
        source_frame_idx = 0
        pending_frames = deque()
        H_init = None

    prev_gray_nm = cv2.cvtColor(first_small, cv2.COLOR_BGR2GRAY)
    max_corners_nm = 1200
    quality_nm = 0.01
    min_dist_nm = 7
    lk_nm = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    prev_pts_nm = cv2.goodFeaturesToTrack(prev_gray_nm, 500, 0.01, 7)
    flow_ema = np.array([0.0, 0.0], dtype=float)
    flow_alpha = 0.2

    marker_half_nav = 0.5
    dst_marker_nav_m = np.array(
        [
            [-marker_half_nav, -marker_half_nav],
            [marker_half_nav, -marker_half_nav],
            [marker_half_nav, marker_half_nav],
            [-marker_half_nav, marker_half_nav],
        ],
        dtype=np.float32,
    )
    track_x = 0.5 * (MARKER_RESIZE_W - 1)
    track_y = 0.85 * (MARKER_RESIZE_H - 1)

    prev_gray_marker = preprocess_gray(first_marker)
    prev_roi_marker = make_roi_mask(prev_gray_marker, ROI_TOP_RATIO)

    def detect_features(gray, mask):
        return cv2.goodFeaturesToTrack(
            gray,
            maxCorners=MAX_CORNERS_MARKER,
            qualityLevel=GFTT_QUALITY,
            minDistance=GFTT_MIN_DIST,
            blockSize=GFTT_BLOCK,
            mask=mask,
        )

    prev_pts_marker = detect_features(prev_gray_marker, prev_roi_marker)
    H_prev_to_plane = H_init
    if H_prev_to_plane is None:
        pts0, _ = detect_red_marker_corners(first_marker)
        if pts0 is not None and polygon_area(pts0) > MARKER_AREA_MIN:
            H_prev_to_plane = cv2.getPerspectiveTransform(pts0.astype(np.float32), dst_marker_nav_m).astype(np.float64)
    last_xy = project_point(H_prev_to_plane, track_x, track_y) if H_prev_to_plane is not None else None
    if last_xy is None:
        last_xy = (0.0, 0.0)
    last_accept_t_abs = source_frame_idx / max(fps, 1e-6)
    H_guess_prev_to_cur = None

    alt_agl_state = 1.5
    alt_prev_gray = cv2.cvtColor(first_marker, cv2.COLOR_BGR2GRAY)
    fx_alt = 1100.0 * (float(MARKER_RESIZE_W) / 853.0)
    fy_alt = 1100.0 * (float(MARKER_RESIZE_H) / 480.0)
    K_alt = np.array(
        [
            [fx_alt, 0.0, float(MARKER_RESIZE_W) / 2.0],
            [0.0, fy_alt, float(MARKER_RESIZE_H) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    alt_prev_pts = cv2.goodFeaturesToTrack(alt_prev_gray, 500, 0.01, 7)
    alt_prev_scale = compute_scale_from_samples(alt_prev_pts.reshape(-1, 2)) if alt_prev_pts is not None else None
    alt_marker_seen = True
    alt_marker_cnt = 0
    lk_alt = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    lk_win = LK_WIN_MARKER if (LK_WIN_MARKER % 2 == 1) else (LK_WIN_MARKER + 1)
    lk_params = dict(
        winSize=(lk_win, lk_win),
        maxLevel=LK_LEVELS_MARKER,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    processed = 0
    if nav_mode == "marker":
        x_nav0 = float(-last_xy[0] if FLIP_X_MARKER else last_xy[0])
        y_nav0 = float(-last_xy[1] if FLIP_Y_MARKER else last_xy[1])
        pos[0] = float(x_nav0 * MARKER_SIZE_XY_M)
        pos[1] = float(y_nav0 * MARKER_SIZE_XY_M)
        pos[2] = float(alt_agl_state)
        if force_fixed_height_frames:
            pos[2] = fixed_height_m
        traj_points[0] = pos.copy()

    while True:
        frame_start = time.perf_counter()
        if sessions.get(session_id, {}).get("stop"):
            break

        if frame_idx == 1:
            frame_orig = first_orig
            frame_small = first_small
        elif pending_frames:
            frame_orig = pending_frames.popleft()
            frame_small = cv2.resize(frame_orig, (NAV_W, NAV_H))
        else:
            ok, frame = run_reader.read()
            if not ok or frame is None:
                break
            frame_orig = frame
            frame_small = cv2.resize(frame_orig, (NAV_W, NAV_H))
        frame_marker = cv2.resize(frame_orig, (MARKER_RESIZE_W, MARKER_RESIZE_H))

        if nav_mode == "marker":
            processed += 1
            t_abs = source_frame_idx / max(fps, 1e-6)
            cur_gray = preprocess_gray(frame_marker)
            cur_roi = make_roi_mask(cur_gray, ROI_TOP_RATIO)

            alt_gray = cv2.cvtColor(frame_marker, cv2.COLOR_BGR2GRAY)
            alt_marker = detect_red_square_corners_alt(frame_marker)
            if alt_marker is not None and polygon_area(alt_marker) > MARKER_AREA_MIN:
                alt_marker_seen = True
                alt_marker_cnt += 1
                cam_pos = estimate_camera_center_from_marker(alt_marker.astype(np.float32), K_alt, 1.0)
                if cam_pos is not None and np.isfinite(cam_pos).all():
                    alt_agl_state = (1.0 - MARKER_ALPHA_ALT) * alt_agl_state + MARKER_ALPHA_ALT * float(cam_pos[2])
                alt_prev_pts = cv2.goodFeaturesToTrack(alt_gray, 500, 0.01, 7)
                alt_prev_scale = compute_scale_from_samples(alt_prev_pts.reshape(-1, 2)) if alt_prev_pts is not None else None
            else:
                if alt_marker_seen and alt_marker_cnt > 10:
                    alt_marker_seen = False
                if (not alt_marker_seen) and alt_prev_pts is not None:
                    nxt, st, _ = cv2.calcOpticalFlowPyrLK(alt_prev_gray, alt_gray, alt_prev_pts, None, **lk_alt)
                    if nxt is not None and st is not None:
                        p0 = alt_prev_pts[st.flatten() == 1].reshape(-1, 2)
                        p1 = nxt[st.flatten() == 1].reshape(-1, 2)
                        if len(p0) > 40:
                            cs = compute_scale_from_samples(p1)
                            if alt_prev_scale is not None and cs is not None:
                                ratio = cs / alt_prev_scale if abs(alt_prev_scale) > 1e-9 else None
                                if ratio is not None and 0.7 < ratio < 1.3:
                                    dh = float(np.clip(alt_agl_state / ratio - alt_agl_state, -MAX_DH_ALT, MAX_DH_ALT))
                                    alt_agl_state = (1.0 - Z_ALPHA_ALT) * alt_agl_state + Z_ALPHA_ALT * (alt_agl_state + dh * PX_TO_M_Z_ALT)
                            if cs is not None:
                                alt_prev_scale = cs
                alt_prev_pts = cv2.goodFeaturesToTrack(alt_gray, 500, 0.01, 7)
            alt_prev_gray = alt_gray

            force_redetect_next = False
            reset = 0
            if MARKER_CHECK_EVERY > 0 and (processed % MARKER_CHECK_EVERY == 0):
                pts_m, _ = detect_red_marker_corners(frame_marker)
                if pts_m is not None and polygon_area(pts_m) > MARKER_AREA_MIN:
                    H_direct = cv2.getPerspectiveTransform(pts_m.astype(np.float32), dst_marker_nav_m).astype(np.float64)
                    xy_direct = project_point(H_direct, track_x, track_y)
                    if xy_direct is not None:
                        dist = float(np.hypot(xy_direct[0] - last_xy[0], xy_direct[1] - last_xy[1]))
                        if dist < 30.0:
                            H_prev_to_plane = H_direct
                            reset = 1
                            last_xy = xy_direct
                            last_accept_t_abs = t_abs

            accepted = False
            if prev_pts_marker is None or len(prev_pts_marker) < MIN_TRACK_PTS_MARKER:
                prev_pts_marker = detect_features(prev_gray_marker, prev_roi_marker)

            if prev_pts_marker is not None and len(prev_pts_marker) >= MIN_TRACK_PTS_MARKER:
                lk_flags = 0
                next_init = None
                if H_guess_prev_to_cur is not None:
                    try:
                        next_init = cv2.perspectiveTransform(prev_pts_marker, H_guess_prev_to_cur)
                        lk_flags |= cv2.OPTFLOW_USE_INITIAL_FLOW
                    except cv2.error:
                        next_init = None
                        lk_flags = 0

                cur_pts, st_fwd, err_fwd = cv2.calcOpticalFlowPyrLK(
                    prev_gray_marker, cur_gray, prev_pts_marker, next_init, flags=lk_flags, **lk_params
                )
                if cur_pts is not None and st_fwd is not None:
                    st_fwd = st_fwd.reshape(-1).astype(bool)
                    p0 = np.empty((0, 2), dtype=np.float32)
                    p1 = np.empty((0, 2), dtype=np.float32)

                    if np.count_nonzero(st_fwd) >= MIN_TRACK_PTS_MARKER:
                        p0_f = prev_pts_marker[st_fwd]
                        p1_f = cur_pts[st_fwd]
                        good = np.ones((len(p0_f),), dtype=bool)

                        if err_fwd is not None and LK_ERR_THR_MARKER > 0:
                            ef = err_fwd.reshape(-1)[st_fwd]
                            good &= np.isfinite(ef) & (ef < LK_ERR_THR_MARKER)

                        if FB_THR_MARKER > 0:
                            back_pts, st_back, err_back = cv2.calcOpticalFlowPyrLK(
                                cur_gray, prev_gray_marker, p1_f, None, **lk_params
                            )
                            if back_pts is not None and st_back is not None:
                                st_back = st_back.reshape(-1).astype(bool)
                                good &= st_back

                                fb = np.linalg.norm(p0_f.reshape(-1, 2) - back_pts.reshape(-1, 2), axis=1)
                                good &= np.isfinite(fb) & (fb < FB_THR_MARKER)

                                if err_back is not None and LK_ERR_THR_MARKER > 0:
                                    eb = err_back.reshape(-1)
                                    good &= np.isfinite(eb) & (eb < LK_ERR_THR_MARKER)

                        p0 = p0_f.reshape(-1, 2)[good]
                        p1 = p1_f.reshape(-1, 2)[good]

                        if REDETECT_MIN_PTS_MARKER > 0 and len(p0) < REDETECT_MIN_PTS_MARKER:
                            force_redetect_next = True

                    if len(p0) >= MIN_TRACK_PTS_MARKER:
                        H_prev_to_cur, inl = cv2.findHomography(
                            p0.reshape(-1, 1, 2),
                            p1.reshape(-1, 1, 2),
                            cv2.RANSAC,
                            RANSAC_THR_MARKER,
                        )
                        if H_prev_to_cur is not None and inl is not None:
                            inliers_cnt = int(inl.sum())
                            ratio = float(inliers_cnt) / float(max(1, len(p0)))
                            if H_prev_to_cur is not None and inliers_cnt >= MIN_INLIERS_MARKER and ratio >= MIN_INLIER_RATIO_MARKER:
                                H_guess_prev_to_cur = H_prev_to_cur
                            min_inl_dyn = max(MIN_INLIERS_MARKER, int(0.30 * len(p0)))

                            if ratio < HARD_RESET_RATIO_MARKER:
                                force_redetect_next = True

                            if inliers_cnt >= min_inl_dyn and ratio >= MIN_INLIER_RATIO_MARKER and H_prev_to_plane is not None:
                                invH = safe_inv_homography(H_prev_to_cur)
                                if invH is not None:
                                    H_cur_to_plane = (H_prev_to_plane @ invH).astype(np.float64)
                                    inl_mask = inl.reshape(-1).astype(bool)
                                    p1_in = p1[inl_mask]
                                    xy_candidate = project_points_median(H_cur_to_plane, p1_in)
                                    if xy_candidate is None:
                                        xy_candidate = project_point(H_cur_to_plane, track_x, track_y)

                                    if xy_candidate is not None:
                                        dt = float(t_abs - last_accept_t_abs)
                                        if dt <= 0:
                                            dt = 1.0 / max(float(fps), 1e-6)
                                        step = float(np.hypot(xy_candidate[0] - last_xy[0], xy_candidate[1] - last_xy[1]))
                                        max_step = MAX_SPEED_MARKER * dt * float(MAX_STEP_SCALE_MARKER)
                                        if reset == 1 or step <= max_step:
                                            accepted = True
                                            last_accept_t_abs = float(t_abs)
                                            H_prev_to_plane = H_cur_to_plane
                                            last_xy = (float(xy_candidate[0]), float(xy_candidate[1]))
                                            prev_pts_marker = p1_in.reshape(-1, 1, 2).astype(np.float32)

            if not accepted:
                prev_pts_marker = detect_features(cur_gray, cur_roi)

            prev_gray_marker = cur_gray
            prev_roi_marker = cur_roi
            if force_redetect_next:
                prev_pts_marker = detect_features(prev_gray_marker, prev_roi_marker)

            x_nav = float(-last_xy[0] if FLIP_X_MARKER else last_xy[0])
            y_nav = float(-last_xy[1] if FLIP_Y_MARKER else last_xy[1])
            pos[0] = float(x_nav * MARKER_SIZE_XY_M)
            pos[1] = float(y_nav * MARKER_SIZE_XY_M)
            pos[2] = float(alt_agl_state)
        else:
            gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
            dx = dy = 0.0
            got_shift = False
            if prev_pts_nm is None or len(prev_pts_nm) < 50:
                prev_pts_nm = cv2.goodFeaturesToTrack(prev_gray_nm, max_corners_nm, quality_nm, min_dist_nm)

            if prev_pts_nm is not None:
                nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray_nm, gray, prev_pts_nm, None, **lk_nm)
                if nxt is not None and st is not None:
                    p0 = prev_pts_nm[st.flatten() == 1].reshape(-1, 2)
                    p1 = nxt[st.flatten() == 1].reshape(-1, 2)
                    if len(p0) >= 20:
                        deltas = p1 - p0
                        dx = float(np.median(deltas[:, 0]))
                        dy = float(np.median(deltas[:, 1]))
                        got_shift = True
                        prev_pts_nm = p1.reshape(-1, 1, 2)
                    else:
                        prev_pts_nm = cv2.goodFeaturesToTrack(gray, max_corners_nm, quality_nm, min_dist_nm)

            if not got_shift:
                shift, _ = cv2.phaseCorrelate(np.float32(prev_gray_nm), np.float32(gray))
                dx = float(shift[0])
                dy = float(shift[1])

            flow_ema = (1.0 - flow_alpha) * flow_ema + flow_alpha * np.array([dx, dy], dtype=float)
            step = np.array([-flow_ema[0], -flow_ema[1]], dtype=float)
            step_len = float(np.linalg.norm(step))
            if step_len > 80:
                step *= 80.0 / step_len
            pos[:2] = pos[:2] + step
            prev_gray_nm = gray

        if force_fixed_height_frames:
            pos[2] = fixed_height_m

        if nav_mode != "marker" and traj_points:
            prev = traj_points[-1]
            jump_xy = np.linalg.norm(pos[:2] - prev[:2])
            jump_z = abs(pos[2] - prev[2])
            if jump_xy > SMOOTH_JUMP_XY or jump_z > SMOOTH_JUMP_Z:
                pos = _laplacian_smooth_last(traj_points, pos)

        detections: List[Dict] = []
        person_count = 0
        det_latency = 0.0
        should_detect = detection_client is not None and ((frame_idx - 1) % profile.detection_stride == 0)
        if should_detect:
            detect_frame = frame_orig if mode_label.endswith(":frames") else frame_small
            raw_detections, _raw_count, det_latency = detection_client.detect(
                detect_frame,
                model=detector_name,
                conf=detector_conf,
                iou=detector_iou,
                max_det=detector_max_det,
            )
            if debug_evaluator is not None:
                frame_path = getattr(run_reader, "last_path", None)
                if frame_path is None:
                    frame_path = getattr(reader, "last_path", None)
                debug_evaluator.update(frame_path=frame_path, frame_shape=detect_frame.shape[:2], detections=raw_detections)
            if display_conf is not None:
                detections = [
                    d
                    for d in raw_detections
                    if float(d.get("conf", d.get("score", 0.0))) >= float(display_conf)
                ]
            else:
                detections = raw_detections
            detections = _scale_detections_xyxy(detections, detect_frame.shape[:2], frame_small.shape[:2])
            person_count = len(detections)
            if detections:
                draw_detections(frame_small, detections)

        if save_video:
            if mode_label.endswith(":frames"):
                if alert_frames_dir is None:
                    alert_frames_dir = REPORT_DIR / f"{report_stem}_alerts"
                    alert_frames_dir.mkdir(parents=True, exist_ok=True)
                if detections:
                    frame_path = alert_frames_dir / f"frame_{frame_idx:06d}.jpg"
                    cv2.imwrite(str(frame_path), frame_small)
                    alert_saved_count += 1
            else:
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_out_path = REPORT_DIR / f"{report_stem}_processed.mp4"
                    writer = cv2.VideoWriter(str(video_out_path), fourcc, max(10.0, min(60.0, fps)), (NAV_W, NAV_H))
                writer.write(frame_small)

        duration = max(time.perf_counter() - t_start, 1e-6)
        timestamp = duration
        fps_display = frame_idx / duration

        traj_points.append(pos.copy())
        time_stamps.append(timestamp)
        traj_np = np.vstack(traj_points)

        payload = {
            "frame_index": frame_idx,
            "timestamp": timestamp,
            "fps": fps_display,
            "video_fps": fps,
            "person_count": person_count,
            "detector_latency_ms": det_latency,
            "scale": None,
            "plane_ready": False,
            "point": {
                "x": float(traj_np[-1][0]),
                "y": float(traj_np[-1][1]),
                "z": float(traj_np[-1][2]),
                "t": float(timestamp),
            },
        }
        if remote_stats_getter is not None:
            now = time.monotonic()
            if now - last_remote_stats_poll >= 2.0:
                last_remote_stats_poll = now
                remote_stats = remote_stats_getter()
            if isinstance(remote_stats, dict):
                payload["stream_stats"] = remote_stats

        fps_samples.append(payload["fps"])
        person_samples.append(person_count)
        det_latency_samples.append(det_latency)

        emit = True
        if profile.emit_only_detections:
            emit = bool(detections)

        if emit:
            payload["frame"] = encode_image_b64(frame_small)
            if traj_np.shape[0] > 1 and (frame_idx % 10 == 0):
                payload["plots"] = render_plots(traj_np, time_stamps)
            on_update(payload)

        frame_idx += 1
        source_frame_idx += 1

        if target_fps is not None and target_fps > 0:
            elapsed = time.perf_counter() - frame_start
            sleep_for = (1.0 / target_fps) - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

    run_reader.close()
    if writer is not None:
        writer.release()

    duration = max(time.perf_counter() - t_start, 1e-6)
    report_url = None
    if traj_points:
        traj_np = np.vstack(traj_points)
        zip_path = save_report(
            report_stem,
            traj_np,
            time_stamps,
            person_samples,
            fps_samples,
            det_latency_samples,
            mode_label,
            video_out_path,
            alert_frames_dir,
        )
        if session_id in sessions:
            sessions[session_id]["report"] = zip_path
        report_url = f"/report/{session_id}"

    summary = {
        "done": True,
        "frames_total": max(0, len(traj_points) - 1),
        "avg_fps": float(max(0, len(traj_points) - 1) / duration),
        "avg_person_count": float(np.mean(person_samples)) if person_samples else 0.0,
        "duration_sec": duration,
        "report_url": report_url,
        "debug_metrics_info": debug_metrics_info,
        "alerts_saved": int(alert_saved_count),
    }
    if debug_evaluator is not None:
        summary["debug_metrics"] = debug_evaluator.summary()
    else:
        summary["debug_metrics"] = None
    if remote_stats_getter is not None:
        final_stats = remote_stats_getter()
        if isinstance(final_stats, dict):
            summary["stream_stats"] = final_stats
    on_update(summary)


def _build_reader(meta: Dict) -> Tuple[object, Optional[Tuple[str, str]]]:
    source_kind = meta["source_kind"]
    nsu_channel = meta["nsu_channel"]
    loop_input = bool(meta.get("loop", False))

    if nsu_channel == "stream":
        remote_session, stream_url, rpi_base, remote_meta = start_remote_source(
            meta.get("rpi_url", ""),
            source_kind,
            meta["source"],
            loop_input=loop_input,
            mission_id=str(meta.get("stream_mission_id", "")).strip(),
            realtime=bool(meta.get("stream_realtime", True)),
            target_fps=meta.get("stream_target_fps"),
            jitter_ms=int(meta.get("stream_jitter_ms", 0) or 0),
            drop_if_lag=bool(meta.get("stream_drop_if_lag", True)),
            max_duration_sec=meta.get("stream_max_duration_sec"),
            jpeg_quality=int(meta.get("stream_jpeg_quality", 80) or 80),
        )
        reader = OpenCVSource(stream_url, "rtsp", loop_input=loop_input)
        if not reader.is_open():
            raise RuntimeError("Не удалось открыть поток с RaspberryPi")
        if isinstance(remote_meta, dict):
            meta["stream_mission_id"] = str(remote_meta.get("mission_id", meta.get("stream_mission_id", "")))
            meta["stream_target_fps"] = float(remote_meta.get("target_fps", meta.get("stream_target_fps") or 0.0))
        return reader, (rpi_base, remote_session)

    if source_kind == "frames":
        reader = FolderSource(meta["source"], loop_input=loop_input)
    else:
        mode = "rtsp" if source_kind == "rtsp" else "file"
        reader = OpenCVSource(meta["source"], mode=mode, loop_input=loop_input)
    if not reader.is_open():
        raise RuntimeError("Не удалось открыть источник")
    return reader, None


app = FastAPI(
    title="Unified Navigation Service",
    description="Единый сервис с режимами НСУ локальный/потоковый и Edge-заглушкой",
)
sessions: Dict[str, Dict] = {}
detection_client = DetectionClient(DETECTION_URL)


@app.on_event("startup")
def startup_event():
    cfg = _load_config()
    _apply_mode_flags(cfg.get("mode_flags", {}))
    app.state.config = cfg
    app.state.detection_client = DetectionClient(cfg.get("detection_url", DETECTION_URL))


@app.get("/")
def index():
    cfg = getattr(app.state, "config", {}) or {}
    ui_template_path = Path(str(cfg.get("ui_template_path", UI_TEMPLATE_PATH)))
    if not ui_template_path.exists():
        raise HTTPException(status_code=500, detail=f"UI template not found: {ui_template_path}")
    return HTMLResponse(ui_template_path.read_text(encoding="utf-8"))


@app.get("/health")
def health():
    det = getattr(app.state, "detection_client", detection_client)
    d_health = det.health()
    return {
        "status": "ok",
        "detection_url": det.base_url,
        "detection": d_health,
        "modes": MODE_FLAGS,
        "debug_presets": sorted(list(DEBUG_PRESET_PATHS.keys())),
        "active_sessions": sum(1 for s in sessions.values() if not s.get("stop", False)),
    }


@app.get("/debug/presets")
def debug_presets():
    names = sorted(list(DEBUG_PRESET_PATHS.keys()))
    out = []
    for name in names:
        try:
            out.append(_load_debug_preset(name))
        except Exception as exc:  # noqa: BLE001
            out.append({"name": name, "error": str(exc)})
    return {"presets": out}


@app.get("/modes")
def modes_state():
    return {"flags": MODE_FLAGS}


@app.get("/report/{session_id}")
def download_report(session_id: str):
    meta = sessions.get(session_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    zip_path = meta.get("report")
    if not zip_path or not Path(zip_path).exists():
        raise HTTPException(status_code=404, detail="Отчет еще не готов")
    filename = Path(zip_path).name
    return FileResponse(path=zip_path, filename=filename, media_type="application/zip")


@app.post("/stop/{session_id}")
async def stop_session(session_id: str):
    meta = sessions.get(session_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    meta["stop"] = True
    remote_link = meta.get("remote_link")
    if remote_link:
        stop_remote_source(remote_link[0], remote_link[1])
    return {"status": "stopping"}


@app.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    detect: str = Form("true"),
    device: str = Form("CPU"),
    save_video: str = Form("false"),
    demo_loop: str = Form("false"),
    marker_mode: str = Form("auto"),
    model: str = Form("yolov8n_baseline_multiscale"),
    run_mode: str = Form("nsu"),
    nsu_channel: str = Form("local"),
):
    _validate_mode_chain(run_mode, nsu_channel, "video")
    if not file.filename:
        raise HTTPException(status_code=400, detail="Файл не передан")

    detect_enabled = str(detect).lower() in ("1", "true", "yes", "on")
    save_video_flag = str(save_video).lower() in ("1", "true", "yes", "on")
    demo_loop_flag = str(demo_loop).lower() in ("1", "true", "yes", "on")

    suffix = Path(file.filename).suffix or ".mp4"
    session_id = uuid.uuid4().hex
    out_path = UPLOAD_DIR / f"{session_id}{suffix}"
    out_path.write_bytes(await file.read())

    created_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_stem = f"{created_ts}_{device.replace(' ', '_')}_{session_id}"

    detector_name = _resolve_detector(model) if run_mode == "nsu" else "yolo"
    detector_conf = None
    detector_iou = None
    detector_max_det = None
    display_conf = None
    target_fps = None
    debug_preset_name = ""
    debug_preset_meta = None
    if run_mode == "nsu" and nsu_channel == "local" and model != "nanodet":
        debug_preset_name = NSU_LOCAL_VIDEO_PRESET
        debug_preset_meta = _load_debug_preset(debug_preset_name)
        model = debug_preset_meta["model_ui"]
        detector_name = _resolve_detector(model)
        detector_conf = float(debug_preset_meta.get("infer_conf", debug_preset_meta["detector_conf"]))
        detector_iou = float(debug_preset_meta.get("infer_nms_iou", 0.7))
        detector_max_det = int(debug_preset_meta.get("infer_max_det", 300))
        display_conf = float(debug_preset_meta.get("display_conf", detector_conf))
        target_fps = debug_preset_meta["target_fps"]

    sessions[session_id] = {
        "run_mode": run_mode,
        "nsu_channel": nsu_channel,
        "source_kind": "video",
        "source": str(out_path),
        "detect": detect_enabled,
        "loop": False,
        "stop": False,
        "delete_after": not demo_loop_flag,
        "device": device,
        "save_video": save_video_flag,
        "demo_loop": demo_loop_flag,
        "created_ts": created_ts,
        "report_stem": report_stem,
        "model_ui": model,
        "model": detector_name,
        "detector_conf": detector_conf,
        "detector_iou": detector_iou,
        "detector_max_det": detector_max_det,
        "display_conf": display_conf,
        "target_fps": target_fps,
        "debug_preset": debug_preset_name,
        "debug_preset_meta": debug_preset_meta,
        "marker_mode": marker_mode,
        "remote_link": None,
    }
    return {
        "session_id": session_id,
        "detect": detect_enabled,
        "mode": "video",
        "applied_config": debug_preset_name,
        "demo_loop": demo_loop_flag,
    }


@app.post("/start_rtsp")
async def start_rtsp(
    rtsp_url: str = Form(...),
    detect: str = Form("true"),
    device: str = Form("CPU"),
    save_video: str = Form("false"),
    marker_mode: str = Form("auto"),
    model: str = Form("yolov8n_baseline_multiscale"),
    run_mode: str = Form("nsu"),
    nsu_channel: str = Form("local"),
    rpi_url: str = Form(""),
):
    source_kind = "rtsp"
    _validate_mode_chain(run_mode, nsu_channel, source_kind)
    _ensure_rtsp_url(rtsp_url)

    detect_enabled = str(detect).lower() in ("1", "true", "yes", "on")
    save_video_flag = str(save_video).lower() in ("1", "true", "yes", "on")
    session_id = uuid.uuid4().hex

    created_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_stem = f"{created_ts}_{device.replace(' ', '_')}_{session_id}"
    detector_name = _resolve_detector(model) if run_mode == "nsu" else "yolo"
    detector_conf = None
    detector_iou = None
    detector_max_det = None
    display_conf = None
    target_fps = None
    debug_preset_name = ""
    debug_preset_meta = None
    if run_mode == "nsu" and nsu_channel == "local" and model != "nanodet":
        debug_preset_name = NSU_LOCAL_VIDEO_PRESET
        debug_preset_meta = _load_debug_preset(debug_preset_name)
        model = debug_preset_meta["model_ui"]
        detector_name = _resolve_detector(model)
        detector_conf = float(debug_preset_meta.get("infer_conf", debug_preset_meta["detector_conf"]))
        detector_iou = float(debug_preset_meta.get("infer_nms_iou", 0.7))
        detector_max_det = int(debug_preset_meta.get("infer_max_det", 300))
        display_conf = float(debug_preset_meta.get("display_conf", detector_conf))
        target_fps = debug_preset_meta["target_fps"]

    source = rtsp_url
    if run_mode == "nsu" and nsu_channel == "stream":
        source = rtsp_url

    sessions[session_id] = {
        "run_mode": run_mode,
        "nsu_channel": nsu_channel,
        "source_kind": source_kind,
        "source": source,
        "rpi_url": rpi_url,
        "detect": detect_enabled,
        "loop": False,
        "stop": False,
        "delete_after": False,
        "device": device,
        "save_video": save_video_flag,
        "created_ts": created_ts,
        "report_stem": report_stem,
        "model_ui": model,
        "model": detector_name,
        "detector_conf": detector_conf,
        "detector_iou": detector_iou,
        "detector_max_det": detector_max_det,
        "display_conf": display_conf,
        "target_fps": target_fps,
        "debug_preset": debug_preset_name,
        "debug_preset_meta": debug_preset_meta,
        "marker_mode": marker_mode,
        "remote_link": None,
    }
    return {
        "session_id": session_id,
        "detect": detect_enabled,
        "mode": "rtsp",
        "applied_config": debug_preset_name,
    }


@app.post("/start_frames")
async def start_frames(
    frames_dir: str = Form(...),
    detect: str = Form("true"),
    device: str = Form("CPU"),
    save_video: str = Form("false"),
    marker_mode: str = Form("auto"),
    model: str = Form("yolov8n_baseline_multiscale"),
    run_mode: str = Form("nsu"),
    nsu_channel: str = Form("local"),
    rpi_url: str = Form(""),
    annotations_dir: str = Form(""),
):
    _validate_mode_chain(run_mode, nsu_channel, "frames")
    detect_enabled = str(detect).lower() in ("1", "true", "yes", "on")
    save_video_flag = str(save_video).lower() in ("1", "true", "yes", "on")

    if run_mode == "nsu" and nsu_channel == "local":
        p = Path(frames_dir)
        if not p.exists() or not p.is_dir():
            raise HTTPException(status_code=400, detail=f"Папка кадров не найдена: {frames_dir}")
        ann_val = annotations_dir.strip()
        if not ann_val:
            ann_val = _autodetect_annotations_dir(p)
        if ann_val:
            ap = Path(ann_val)
            if not ap.exists() or not ap.is_dir():
                raise HTTPException(status_code=400, detail=f"Папка аннотаций не найдена: {ann_val}")
            if not any(ap.rglob("*.txt")):
                raise HTTPException(status_code=400, detail=f"В папке аннотаций нет .txt файлов: {ann_val}")

    session_id = uuid.uuid4().hex
    created_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_stem = f"{created_ts}_{device.replace(' ', '_')}_{session_id}"
    detector_name = _resolve_detector(model) if run_mode == "nsu" else "yolo"

    sessions[session_id] = {
        "run_mode": run_mode,
        "nsu_channel": nsu_channel,
        "source_kind": "frames",
        "source": frames_dir,
        "rpi_url": rpi_url,
        "detect": detect_enabled,
        "loop": False,
        "stop": False,
        "delete_after": False,
        "device": device,
        "save_video": save_video_flag,
        "created_ts": created_ts,
        "report_stem": report_stem,
        "model_ui": model,
        "model": detector_name,
        "marker_mode": marker_mode,
        "remote_link": None,
        "annotations_dir": annotations_dir.strip() or _autodetect_annotations_dir(Path(frames_dir)),
    }
    return {
        "session_id": session_id,
        "detect": detect_enabled,
        "mode": "frames",
        "annotations_dir": sessions[session_id].get("annotations_dir", ""),
    }


@app.post("/start_source")
async def start_source(
    source_kind: str = Form(...),
    source_value: str = Form(...),
    detect: str = Form("true"),
    device: str = Form("CPU"),
    save_video: str = Form("false"),
    marker_mode: str = Form("auto"),
    model: str = Form("yolov8n_baseline_multiscale"),
    run_mode: str = Form("nsu"),
    nsu_channel: str = Form("local"),
    rpi_url: str = Form(""),
    annotations_dir: str = Form(""),
    annotations_coco: str = Form(""),
    annotations_file: Optional[UploadFile] = File(None),
    debug_preset: str = Form(""),
    stream_mission_id: str = Form(""),
    stream_realtime: str = Form("true"),
    stream_target_fps: str = Form("0"),
    stream_jitter_ms: str = Form("0"),
    stream_drop_if_lag: str = Form("true"),
    stream_max_duration_sec: str = Form("0"),
    stream_jpeg_quality: str = Form("80"),
):
    if source_kind not in {"video", "rtsp", "frames"}:
        raise HTTPException(status_code=400, detail="source_kind должен быть video|rtsp|frames")
    _validate_mode_chain(run_mode, nsu_channel, source_kind)
    if source_kind == "rtsp":
        _ensure_rtsp_url(source_value)

    detect_enabled = str(detect).lower() in ("1", "true", "yes", "on")
    save_video_flag = str(save_video).lower() in ("1", "true", "yes", "on")
    stream_realtime_flag = str(stream_realtime).lower() in ("1", "true", "yes", "on")
    stream_drop_if_lag_flag = str(stream_drop_if_lag).lower() in ("1", "true", "yes", "on")
    try:
        stream_target_fps_val = max(0.0, float(stream_target_fps))
    except Exception:
        stream_target_fps_val = 0.0
    try:
        stream_jitter_ms_val = max(0, int(float(stream_jitter_ms)))
    except Exception:
        stream_jitter_ms_val = 0
    try:
        stream_max_duration_sec_val = max(0.0, float(stream_max_duration_sec))
    except Exception:
        stream_max_duration_sec_val = 0.0
    try:
        stream_jpeg_quality_val = max(10, min(100, int(float(stream_jpeg_quality))))
    except Exception:
        stream_jpeg_quality_val = 80

    annotations_value = annotations_dir.strip()
    annotations_coco_value = annotations_coco.strip()
    forced_frames_preset = "nsu_frames_yolov8n_alert_contract"
    forced_video_preset = NSU_LOCAL_VIDEO_PRESET
    if source_kind == "frames":
        # В режиме потока кадров фиксируем единый боевой конфиг.
        debug_preset = forced_frames_preset
        if run_mode == "nsu" and nsu_channel == "local":
            if not source_value.strip():
                source_value = str(NSU_LOCAL_FRAMES_DIR)
            if not annotations_coco_value.strip():
                annotations_coco_value = str(NSU_LOCAL_FRAMES_COCO)
    elif source_kind in {"video", "rtsp"} and run_mode == "nsu" and nsu_channel == "local" and model != "nanodet":
        debug_preset = forced_video_preset
        annotations_value = ""
    if run_mode == "nsu" and nsu_channel == "local":
        if source_kind == "frames":
            p = Path(source_value)
            if not p.exists() or not p.is_dir():
                raise HTTPException(status_code=400, detail=f"Папка кадров не найдена: {source_value}")
            image_files = FolderSource._scan_image_files(p)
            if not image_files:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"В папке не найдено изображений (.jpg/.jpeg/.png): {source_value}. "
                        "Проверьте путь и вложенные подпапки."
                    ),
                )
        if source_kind in {"video", "rtsp"} and not source_value:
            raise HTTPException(status_code=400, detail="source_value не может быть пустым")
        if source_kind == "frames" and annotations_file is None:
            coco_path = Path(annotations_coco_value)
            if not coco_path.exists() or not coco_path.is_file():
                raise HTTPException(status_code=400, detail=f"COCO аннотации не найдены: {annotations_coco_value}")

    session_id = uuid.uuid4().hex
    created_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_stem = f"{created_ts}_{device.replace(' ', '_')}_{session_id}"
    detector_name = _resolve_detector(model) if run_mode == "nsu" else "yolo"
    detector_conf = None
    detector_iou = None
    detector_max_det = None
    display_conf = None
    target_fps = None
    debug_preset_name = debug_preset.strip()
    debug_preset_meta = None
    if debug_preset_name:
        if run_mode != "nsu":
            raise HTTPException(status_code=400, detail="debug_preset доступен только для run_mode=nsu")
        debug_preset_meta = _load_debug_preset(debug_preset_name)
        model = debug_preset_meta["model_ui"]
        detector_name = _resolve_detector(model)
        detector_conf = float(debug_preset_meta.get("infer_conf", debug_preset_meta["detector_conf"]))
        detector_iou = float(debug_preset_meta.get("infer_nms_iou", 0.7))
        detector_max_det = int(debug_preset_meta.get("infer_max_det", 300))
        display_conf = float(debug_preset_meta.get("display_conf", detector_conf))
        target_fps = debug_preset_meta["target_fps"]

    if source_kind == "frames" and annotations_file is not None:
        filename = annotations_file.filename or ""
        if not filename.lower().endswith(".json"):
            raise HTTPException(status_code=400, detail="Файл аннотаций должен быть .json")
        uploaded_path = UPLOAD_DIR / f"{session_id}_coco.json"
        uploaded_path.write_bytes(await annotations_file.read())
        annotations_coco_value = str(uploaded_path)

    sessions[session_id] = {
        "run_mode": run_mode,
        "nsu_channel": nsu_channel,
        "source_kind": source_kind,
        "source": source_value,
        "rpi_url": rpi_url,
        "detect": detect_enabled,
        "loop": False,
        "stop": False,
        "delete_after": False,
        "device": device,
        "save_video": save_video_flag,
        "created_ts": created_ts,
        "report_stem": report_stem,
        "model_ui": model,
        "model": detector_name,
        "detector_conf": detector_conf,
        "detector_iou": detector_iou,
        "detector_max_det": detector_max_det,
        "display_conf": display_conf,
        "target_fps": target_fps,
        "debug_preset": debug_preset_name,
        "debug_preset_meta": debug_preset_meta,
        "marker_mode": marker_mode,
        "remote_link": None,
        "annotations_dir": annotations_value,
        "coco_gt_override": annotations_coco_value if source_kind == "frames" else "",
        "stream_mission_id": stream_mission_id.strip(),
        "stream_realtime": stream_realtime_flag,
        "stream_target_fps": stream_target_fps_val,
        "stream_jitter_ms": stream_jitter_ms_val,
        "stream_drop_if_lag": stream_drop_if_lag_flag,
        "stream_max_duration_sec": stream_max_duration_sec_val,
        "stream_jpeg_quality": stream_jpeg_quality_val,
    }
    return {
        "session_id": session_id,
        "detect": detect_enabled,
        "mode": source_kind,
        "applied_config": debug_preset_name if source_kind in {"frames", "video", "rtsp"} else "",
        "stream_mission_id": stream_mission_id.strip(),
    }


@app.websocket("/ws/process/{session_id}")
async def ws_process(websocket: WebSocket, session_id: str):
    await websocket.accept()
    meta = sessions.get(session_id)
    if meta is None:
        await websocket.send_text(json.dumps({"error": "Сессия не найдена"}))
        await websocket.close()
        return

    run_mode = meta.get("run_mode", "nsu")
    nsu_channel = meta.get("nsu_channel", "local")
    source_kind = meta.get("source_kind", "video")
    _validate_mode_chain(run_mode, nsu_channel, source_kind)

    if run_mode == "edge":
        await websocket.send_text(
            json.dumps(
                {
                    "info": "Edge режим выбран, но пока не реализован (зарезервировано для полного инференса на RaspberryPi)",
                    "done": True,
                    "frames_total": 0,
                    "avg_fps": 0.0,
                    "avg_person_count": 0.0,
                    "duration_sec": 0.0,
                    "report_url": None,
                }
            )
        )
        await websocket.close()
        return

    detector_name = meta.get("model", "yolo")
    detector_conf = meta.get("detector_conf")
    if detector_conf is not None:
        try:
            detector_conf = float(detector_conf)
        except Exception:
            detector_conf = None
    detector_iou = meta.get("detector_iou")
    if detector_iou is not None:
        try:
            detector_iou = float(detector_iou)
        except Exception:
            detector_iou = None
    detector_max_det = meta.get("detector_max_det")
    if detector_max_det is not None:
        try:
            detector_max_det = int(detector_max_det)
        except Exception:
            detector_max_det = None
    target_fps = meta.get("target_fps")
    if target_fps is not None:
        try:
            target_fps = float(target_fps)
        except Exception:
            target_fps = None
    display_conf = meta.get("display_conf")
    if display_conf is not None:
        try:
            display_conf = float(display_conf)
        except Exception:
            display_conf = None
    marker_mode = meta.get("marker_mode", "auto")
    detect_enabled = bool(meta.get("detect", True))
    save_video = bool(meta.get("save_video", False))
    report_stem = meta.get("report_stem", f"{session_id}")

    profile = source_profile(source_kind)

    sessions[session_id]["stop"] = False
    loop = asyncio.get_event_loop()

    def push(msg: Dict):
        future = asyncio.run_coroutine_threadsafe(websocket.send_text(json.dumps(msg)), loop)

        def _mark_stop(fut):
            if session_id in sessions:
                try:
                    if fut.cancelled() or fut.exception() is not None:
                        sessions[session_id]["stop"] = True
                except Exception:
                    sessions[session_id]["stop"] = True

        future.add_done_callback(_mark_stop)

    try:
        det_client = getattr(app.state, "detection_client", detection_client)
        if detect_enabled:
            d_health = det_client.health()
            if not d_health.get("_reachable"):
                await websocket.send_text(
                    json.dumps(
                        {
                            "error": (
                                f"Сервис детекции недоступен: {det_client.base_url}. "
                                f"Проверьте запуск services/detection_service.py. "
                                f"Детали: {d_health.get('detail', 'n/a')}"
                            )
                        }
                    )
                )
                return
        reader, remote_link = _build_reader(meta)
        sessions[session_id]["remote_link"] = remote_link
        remote_stats_getter = None
        if remote_link:
            rpi_base, remote_session = remote_link

            def _remote_stats_getter():
                return fetch_remote_source_stats(rpi_base, remote_session)

            remote_stats_getter = _remote_stats_getter
            await websocket.send_text(
                json.dumps(
                    {
                        "stream_info": {
                            "remote_session_id": remote_session,
                            "mission_id": str(meta.get("stream_mission_id", "")),
                            "target_fps": float(meta.get("stream_target_fps") or 0.0),
                            "realtime": bool(meta.get("stream_realtime", True)),
                        }
                    }
                )
            )
        debug_evaluator = None
        debug_metrics_info = ""
        debug_preset_meta = meta.get("debug_preset_meta")
        if source_kind == "frames" and isinstance(debug_preset_meta, dict):
            coco_gt_path = str(meta.get("coco_gt_override", "")).strip() or str(debug_preset_meta.get("dataset_coco_gt", "")).strip()
            images_dir = str(meta.get("source", "")).strip() or str(debug_preset_meta.get("dataset_images_dir", "")).strip()
            if not coco_gt_path:
                debug_metrics_info = "Debug-метрики выключены: в preset не указан dataset.coco_gt."
            else:
                coco_path = Path(coco_gt_path)
                if not coco_path.is_absolute():
                    coco_path = Path.cwd() / coco_path
                images_root = Path(images_dir) if images_dir else None
                if images_root is not None and not images_root.is_absolute():
                    images_root = Path.cwd() / images_root
                debug_evaluator = CocoAlertContractEvaluator(
                    coco_gt_path=coco_path,
                    images_root=images_root,
                    cfg={
                        "thresholds": [float(x) for x in debug_preset_meta.get("eval_thresholds", [])],
                        "target_recall": debug_preset_meta.get("target_recall"),
                        "fp_per_min_target": debug_preset_meta.get("fp_per_min_target"),
                        "fp_total_max": debug_preset_meta.get("fp_total_max", 1.0e12),
                        "fps": debug_preset_meta.get("target_fps"),
                        "alert_contract": dict(debug_preset_meta.get("alert_contract", {})),
                    },
                )
                debug_metrics_info = ""
        elif not str(meta.get("annotations_dir", "")).strip():
            debug_metrics_info = ""
        elif source_kind != "frames" or run_mode != "nsu" or nsu_channel != "local":
            debug_metrics_info = "Debug-метрики доступны только для режима НСУ local + поток кадров."
        elif detector_name not in {"yolo", "yolov8n_baseline_multiscale"}:
            debug_metrics_info = (
                "Debug-метрики считаются только для YOLO-моделей: "
                "yolov8n_baseline_multiscale."
            )
        else:
            debug_metrics_info = ""

        if debug_metrics_info:
            await websocket.send_text(json.dumps({"debug_metrics_info": debug_metrics_info}))
        await loop.run_in_executor(
            None,
            run_unified_pipeline,
            session_id,
            reader,
            det_client if detect_enabled else None,
            push,
            profile,
            marker_mode,
            save_video,
            report_stem,
            detector_name,
            detector_conf,
            detector_iou,
            detector_max_det,
            display_conf,
            target_fps,
            f"{run_mode}:{nsu_channel}:{source_kind}",
            debug_evaluator,
            debug_metrics_info,
            remote_stats_getter,
        )
    except WebSocketDisconnect:
        if session_id in sessions:
            sessions[session_id]["stop"] = True
    except Exception as exc:  # noqa: BLE001
        await websocket.send_text(json.dumps({"error": f"Ошибка: {exc}"}))
    finally:
        if session_id in sessions:
            sessions[session_id]["stop"] = True
            remote_link = sessions[session_id].get("remote_link")
            if remote_link:
                stop_remote_source(remote_link[0], remote_link[1])
            if sessions[session_id].get("delete_after"):
                try:
                    Path(sessions[session_id].get("source", "")).unlink(missing_ok=True)
                except Exception:
                    pass
        await websocket.close()




if __name__ == "__main__":
    import uvicorn # type: ignore

    uvicorn.run("services.unified_runtime.unified_navigation_service:app", host="0.0.0.0", port=8010, reload=False)
