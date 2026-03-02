import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # type: ignore
from pydantic import BaseModel  # type: ignore
from ultralytics import YOLO

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

MODEL_DEFAULT_CONF = {
    "yolov8n_baseline_multiscale": 0.02,
    "yolo": 0.02,
    "nanodet15": 0.35,
    "nanodet15_onnx": 0.35,
    "nanodet": 0.35,
}
MODEL_ALIASES = {
    "nanodet": "nanodet15",
    "nanodet_onnx": "nanodet15_onnx",
    "yolo12": "yolov8n_baseline_multiscale",
}
YOLOV8N_BASELINE_MULTISCALE_WEIGHTS = Path("models/yolov8n_baseline_multiscale.pt")
YOLO_MODEL_WEIGHTS = {
    "yolov8n_baseline_multiscale": YOLOV8N_BASELINE_MULTISCALE_WEIGHTS,
    "yolo": YOLOV8N_BASELINE_MULTISCALE_WEIGHTS,
}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
NANODET_ROOT = PROJECT_ROOT / "nanodet"
NANODET_CONFIG = NANODET_ROOT / "config/nanodet-plus-m-1.5x_416.yml"
NANODET_WEIGHTS = NANODET_ROOT / "nanodet/model/weights/nanodet-plus-m-1.5x_416.pth"
NANODET_ONNX = NANODET_ROOT / "nanodet/model/weights/nanodet-plus-m-1.5x_416.onnx"
DETECTION_CONFIG_PATH = Path("configs/detection_service.yaml")

if NANODET_ROOT.exists():
    nanodet_path = str(NANODET_ROOT.resolve())
    if nanodet_path not in sys.path:
        sys.path.insert(0, nanodet_path)


def _load_detection_config():
    if not DETECTION_CONFIG_PATH.exists() or yaml is None:
        return
    loaded = yaml.safe_load(DETECTION_CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        return

    confs = loaded.get("default_conf_thresholds", {})
    if isinstance(confs, dict):
        for key, val in confs.items():
            try:
                MODEL_DEFAULT_CONF[str(key).strip().lower()] = float(val)
            except Exception:
                continue

    aliases = loaded.get("model_aliases", {})
    if isinstance(aliases, dict):
        for key, val in aliases.items():
            if key is None or val is None:
                continue
            MODEL_ALIASES[str(key).strip().lower()] = str(val).strip().lower()

    yolo_weights_cfg = loaded.get("yolo_weights", {})
    if isinstance(yolo_weights_cfg, dict):
        for key, val in yolo_weights_cfg.items():
            if key is None or val is None:
                continue
            YOLO_MODEL_WEIGHTS[str(key).strip().lower()] = Path(str(val))


_load_detection_config()


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _download_if_missing(target: Path, url: str, *, label: str, timeout_sec: int = 120) -> None:
    if target.exists():
        return
    if not url:
        raise FileNotFoundError(f"Отсутствует {label}: {target} (URL не задан)")
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(target.suffix + ".tmp")
    try:
        with requests.get(url, stream=True, timeout=timeout_sec) as response:
            response.raise_for_status()
            with tmp_path.open("wb") as fd:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        fd.write(chunk)
        tmp_path.replace(target)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def ensure_yolo_assets() -> Path:
    yolo_target = Path(os.getenv("YOLO_WEIGHTS_PATH", str(YOLOV8N_BASELINE_MULTISCALE_WEIGHTS)))
    yolo_url = os.getenv(
        "YOLO_WEIGHTS_URL",
        "https://github.com/ykvnkm/rescueai-models/releases/download/v1/yolov8n_baseline_multiscale.pt",
    )
    _download_if_missing(yolo_target, yolo_url, label="веса YOLO")
    return yolo_target


def ensure_nanodet_assets(require_onnx: bool = False) -> None:
    if env_flag("REQUIRE_NANODET_CONFIG", default=False):
        config_target = Path(os.getenv("NANODET_CONFIG", str(NANODET_CONFIG)))
        config_url = os.getenv("NANODET_CONFIG_URL", "")
        _download_if_missing(config_target, config_url, label="конфиг NanoDet")

    require_pth = env_flag("REQUIRE_NANODET_PTH", default=True) or not require_onnx
    if require_pth:
        pth_target = Path(os.getenv("NANODET_WEIGHTS", str(NANODET_WEIGHTS)))
        pth_url = os.getenv("NANODET_PTH_URL", "")
        _download_if_missing(pth_target, pth_url, label="веса NanoDet (pth)")

    require_onnx_flag = env_flag("REQUIRE_NANODET_ONNX", default=False) or require_onnx
    if require_onnx_flag:
        onnx_target = Path(os.getenv("NANODET_ONNX_PATH", str(NANODET_ONNX)))
        onnx_url = os.getenv("NANODET_ONNX_URL", "")
        _download_if_missing(onnx_target, onnx_url, label="веса NanoDet (onnx)")


def get_device(device_str: str = "auto"):
    device_env = os.getenv("DET_DEVICE")
    return torch.device(
        device_env
        if device_env
        else (device_str if device_str != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    )


def load_yolo(device, weights_path: Path = YOLOV8N_BASELINE_MULTISCALE_WEIGHTS):
    if not weights_path.exists():
        raise FileNotFoundError(f"Не найден файл весов: {weights_path}")
    model = YOLO(str(weights_path))
    model.to(str(device))
    model.fuse()
    return model


def load_nanodet_plus(device):
    ensure_nanodet_assets(require_onnx=False)
    config_path = Path(os.getenv("NANODET_CONFIG", str(NANODET_CONFIG)))
    weights_path = Path(os.getenv("NANODET_WEIGHTS", str(NANODET_WEIGHTS)))
    if not config_path.exists():
        raise FileNotFoundError(f"Не найден конфиг NanoDet: {config_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Не найдены веса NanoDet: {weights_path}")

    from nanodet.data.batch_process import stack_batch_img  # noqa: WPS433
    from nanodet.data.collate import naive_collate  # noqa: WPS433
    from nanodet.data.transform import Pipeline  # noqa: WPS433
    from nanodet.model.arch import build_model  # noqa: WPS433
    from nanodet.util.check_point import load_model_weight  # noqa: WPS433
    from nanodet.util.config import cfg as nanodet_cfg, load_config  # noqa: WPS433

    class _Logger:
        def log(self, msg: str):
            print(msg)

    load_config(nanodet_cfg, str(config_path))
    logger = _Logger()
    model = build_model(nanodet_cfg.model)
    ckpt = torch.load(weights_path, map_location=lambda storage, loc: storage)
    load_model_weight(model, ckpt, logger)

    if nanodet_cfg.model.arch.backbone.name == "RepVGG":
        deploy_config = nanodet_cfg.model
        deploy_config.arch.backbone.update({"deploy": True})
        deploy_model = build_model(deploy_config)
        from nanodet.model.backbone.repvgg import repvgg_det_model_convert  # noqa: WPS433

        model = repvgg_det_model_convert(model, deploy_model)

    model = model.to(device).eval()
    pipeline = Pipeline(nanodet_cfg.data.val.pipeline, nanodet_cfg.data.val.keep_ratio)
    class_names = list(nanodet_cfg.class_names) if nanodet_cfg.class_names else []
    person_idx = class_names.index("person") if "person" in class_names else 0

    return {
        "model": model,
        "device": device,
        "kind": "nanodet15",
        "preprocess": {
            "pipeline": pipeline,
            "input_size": nanodet_cfg.data.val.input_size,
            "stack_batch_img": stack_batch_img,
            "naive_collate": naive_collate,
            "class_names": class_names,
            "person_idx": person_idx,
            "backend": "torch",
        },
    }


def load_nanodet_plus_onnx(device):
    ensure_nanodet_assets(require_onnx=True)
    config_path = Path(os.getenv("NANODET_CONFIG", str(NANODET_CONFIG)))
    onnx_path = Path(os.getenv("NANODET_ONNX_PATH", str(NANODET_ONNX)))
    if not config_path.exists():
        raise FileNotFoundError(f"Не найден конфиг NanoDet: {config_path}")
    if not onnx_path.exists():
        raise FileNotFoundError(f"Не найден ONNX NanoDet: {onnx_path}")

    try:
        import onnxruntime as ort  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "ONNX Runtime не установлен. Установите onnxruntime для запуска NanoDet в ONNX."
        ) from exc

    from nanodet.data.batch_process import stack_batch_img  # noqa: WPS433
    from nanodet.data.collate import naive_collate  # noqa: WPS433
    from nanodet.data.transform import Pipeline  # noqa: WPS433
    from nanodet.model.arch import build_model  # noqa: WPS433
    from nanodet.util.config import cfg as nanodet_cfg, load_config  # noqa: WPS433

    load_config(nanodet_cfg, str(config_path))
    model = build_model(nanodet_cfg.model)
    model.eval()
    pipeline = Pipeline(nanodet_cfg.data.val.pipeline, nanodet_cfg.data.val.keep_ratio)
    class_names = list(nanodet_cfg.class_names) if nanodet_cfg.class_names else []
    person_idx = class_names.index("person") if "person" in class_names else 0

    providers = ["CPUExecutionProvider"]
    available = set(ort.get_available_providers())
    if device.type == "cuda" and "CUDAExecutionProvider" in available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    input_name = session.get_inputs()[0].name

    return {
        "model": session,
        "device": torch.device("cpu"),
        "kind": "nanodet15",
        "preprocess": {
            "pipeline": pipeline,
            "input_size": nanodet_cfg.data.val.input_size,
            "stack_batch_img": stack_batch_img,
            "naive_collate": naive_collate,
            "class_names": class_names,
            "person_idx": person_idx,
            "backend": "onnx",
            "input_name": input_name,
            "head": model.head,
            "num_classes": model.head.num_classes,
        },
    }


def _extract_yolo_person_detections(result, conf_threshold: float):
    names = result.names
    boxes = result.boxes
    if boxes is None:
        return []
    cls = boxes.cls.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy()
    xyxy = boxes.xyxy.cpu().numpy()
    person_id = None
    if isinstance(names, dict):
        for k, v in names.items():
            if str(v).lower() == "person":
                person_id = int(k)
                break
    detections = []
    for box, score, label in zip(xyxy, conf, cls):
        if float(score) < conf_threshold:
            continue
        if person_id is not None and label != person_id:
            continue
        detections.append((box, float(score)))
    return detections


def detect_people(
    frame_bgr: np.ndarray,
    model,
    device: torch.device,
    model_kind: str,
    conf_threshold: float,
    iou_threshold: float | None = None,
    max_det: int | None = None,
    preprocess=None,
    runtime: Optional[dict] = None,
):
    if model_kind == "yolo":
        runtime = runtime or {}
        imgsz = int(runtime.get("imgsz", 640))
        kwargs = {
            "conf": conf_threshold,
            "verbose": False,
            "imgsz": imgsz,
            "device": str(device),
        }
        if iou_threshold is not None:
            kwargs["iou"] = float(iou_threshold)
        if max_det is not None:
            kwargs["max_det"] = int(max_det)
        res = model(frame_bgr, **kwargs)
        if not res:
            return []
        return _extract_yolo_person_detections(res[0], conf_threshold)

    if model_kind == "nanodet15":
        if not isinstance(preprocess, dict):
            raise ValueError("NanoDet preprocess config не задан")
        pipeline = preprocess.get("pipeline")
        input_size = preprocess.get("input_size")
        stack_batch_img = preprocess.get("stack_batch_img")
        naive_collate = preprocess.get("naive_collate")
        person_label = preprocess.get("person_idx", 0)
        backend = preprocess.get("backend", "torch")
        if pipeline is None or input_size is None or stack_batch_img is None or naive_collate is None:
            raise ValueError("NanoDet preprocess config неполный")

        img_info = {"id": 0, "file_name": None, "height": frame_bgr.shape[0], "width": frame_bgr.shape[1]}
        meta = dict(img_info=img_info, raw_img=frame_bgr, img=frame_bgr)
        meta = pipeline(None, meta, input_size)
        meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1)).to(device)
        meta = naive_collate([meta])
        meta["img"] = stack_batch_img(meta["img"], divisible=32)
        if backend == "onnx":
            input_name = preprocess.get("input_name")
            head = preprocess.get("head")
            num_classes = preprocess.get("num_classes")
            if input_name is None or head is None or num_classes is None:
                raise ValueError("NanoDet ONNX preprocess config неполный")
            input_data = meta["img"].cpu().numpy().astype(np.float32, copy=False)
            output = model.run(None, {input_name: input_data})
            if not output:
                return []
            preds = torch.from_numpy(output[0]).float()
            cls_scores = preds[..., :num_classes]
            cls_logits = torch.logit(cls_scores.clamp(1e-6, 1 - 1e-6))
            preds_for_post = torch.cat([cls_logits, preds[..., num_classes:]], dim=-1)
            with torch.no_grad():
                results = head.post_process(preds_for_post, meta)
        else:
            with torch.no_grad():
                results = model.inference(meta)
        dets = results.get(0) if isinstance(results, dict) else None
        if dets is None:
            if isinstance(results, dict) and results:
                dets = next(iter(results.values()))
            else:
                return []
        cls_boxes = dets.get(person_label, []) if isinstance(dets, dict) else []
        detections = []
        for box in cls_boxes:
            if len(box) < 5:
                continue
            x1, y1, x2, y2, score = box[:5]
            if score < conf_threshold:
                continue
            detections.append((np.array([x1, y1, x2, y2], dtype=float), float(score)))
        return detections

    return []


class Detection(BaseModel):
    box: List[float]
    score: float
    stage: Optional[str] = None


class DetectResponse(BaseModel):
    detections: List[Detection]
    person_count: int
    inference_ms: float
    model: str
    used_conf: float


app = FastAPI(title="Detection Service", description="Сервис детекции людей для локального пайплайна")
MODELS: Dict[str, dict] = {}


@app.on_event("startup")
def _load_model():
    device = get_device()
    yolo_weights = ensure_yolo_assets()
    YOLO_MODEL_WEIGHTS["yolov8n_baseline_multiscale"] = yolo_weights
    YOLO_MODEL_WEIGHTS["yolo"] = yolo_weights
    MODELS["yolov8n_baseline_multiscale"] = {
        "model": load_yolo(device, YOLO_MODEL_WEIGHTS["yolov8n_baseline_multiscale"]),
        "device": device,
        "kind": "yolo",
        "preprocess": None,
        "runtime": {"imgsz": 960},
    }
    MODELS["yolo"] = MODELS["yolov8n_baseline_multiscale"]
    app.state.default_model = "yolov8n_baseline_multiscale"
    print(f"[INFO] Модель YOLOv8n Baseline Multiscale загружена на {device}")


@app.get("/health")
def health():
    return {"status": "ok", "models": list(MODELS.keys())}


def get_model(name: str):
    name = name.lower().strip()
    name = MODEL_ALIASES.get(name, name)
    if name in MODELS:
        return MODELS[name]
    device = get_device()
    if name in YOLO_MODEL_WEIGHTS:
        if name in ("yolov8n_baseline_multiscale", "yolo") and not YOLO_MODEL_WEIGHTS[name].exists():
            YOLO_MODEL_WEIGHTS[name] = ensure_yolo_assets()
        MODELS[name] = {
            "model": load_yolo(device, YOLO_MODEL_WEIGHTS[name]),
            "device": device,
            "kind": "yolo",
            "preprocess": None,
            "runtime": {"imgsz": 960},
        }
    elif name in ("nanodet15", "nanodet15_onnx"):
        if name == "nanodet15_onnx" or env_flag("NANODET_ONNX"):
            MODELS[name] = load_nanodet_plus_onnx(device)
        else:
            MODELS[name] = load_nanodet_plus(device)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Неизвестная модель: {name}. Разрешено: yolov8n_baseline_multiscale, nanodet.",
        )
    return MODELS[name]


@app.post("/detect", response_model=DetectResponse)
async def detect(
    file: UploadFile = File(...),
    model: str = Form("yolov8n_baseline_multiscale"),
    conf: str = Form(""),
    iou: str = Form(""),
    max_det: str = Form(""),
):
    data = await file.read()
    np_buf = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Не удалось декодировать изображение")

    requested_model = model.lower().strip()
    resolved_model = MODEL_ALIASES.get(requested_model, requested_model)
    model_entry = get_model(model)
    try:
        conf_val = float(conf) if conf not in (None, "", "null") else None
    except Exception:
        conf_val = None
    try:
        iou_val = float(iou) if iou not in (None, "", "null") else None
    except Exception:
        iou_val = None
    try:
        max_det_val = int(float(max_det)) if max_det not in (None, "", "null") else None
    except Exception:
        max_det_val = None
    if conf_val is None:
        conf_val = MODEL_DEFAULT_CONF.get(
            requested_model,
            MODEL_DEFAULT_CONF.get(resolved_model, MODEL_DEFAULT_CONF.get(model_entry["kind"], 0.5)),
        )

    start = time.perf_counter()
    detections = detect_people(
        frame,
        model_entry["model"],
        model_entry["device"],
        model_entry["kind"],
        conf_val,
        iou_threshold=iou_val,
        max_det=max_det_val,
        preprocess=model_entry.get("preprocess"),
        runtime=model_entry.get("runtime"),
    )
    elapsed = (time.perf_counter() - start) * 1000.0

    out: List[Detection] = []
    for det in detections:
        if len(det) >= 3:
            box, score, stage = det[0], det[1], det[2]
            out.append(Detection(box=[float(x) for x in box], score=float(score), stage=str(stage)))
        else:
            box, score = det
            out.append(Detection(box=[float(x) for x in box], score=float(score)))
    return DetectResponse(
        detections=out,
        person_count=len(out),
        inference_ms=elapsed,
        model=model_entry["kind"],
        used_conf=float(conf_val),
    )


if __name__ == "__main__":
    import uvicorn  # type: ignore

    uvicorn.run("services.detection_service:app", host="0.0.0.0", port=8001, reload=False)
