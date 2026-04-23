from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


NAV_MATCHER_LEGACY = "legacy"
NAV_MATCHER_XFEAT_LIGHTERGLUE = "xfeat_lighterglue"
VALID_NAV_MATCHERS = {NAV_MATCHER_LEGACY, NAV_MATCHER_XFEAT_LIGHTERGLUE}

ACCELERATED_FEATURES_URL = "https://github.com/verlab/accelerated_features.git"
ACCELERATED_FEATURES_HUB = "verlab/accelerated_features"
XFEAT_WEIGHTS_URL = "https://github.com/verlab/accelerated_features/raw/main/weights/xfeat.pt"
DEFAULT_ACCELERATED_FEATURES_REF = "main"
DEFAULT_ACCELERATED_FEATURES_DIR = Path(__file__).resolve().parents[2] / "third_party" / "accelerated_features"


@dataclass(frozen=True)
class MarkerMatcherConfig:
    max_corners: int
    min_track_points: int
    quality_level: float
    min_distance: int
    block_size: int
    lk_win: int
    lk_levels: int
    fb_threshold: float
    lk_error_threshold: float
    redetect_min_points: int
    xfeat_top_k: int = 800
    xfeat_detection_threshold: float = 0.05
    xfeat_min_conf: float = 0.25
    xfeat_max_matches: int = 300
    xfeat_motion_gate_px: float = 35.0
    xfeat_ransac_threshold: float = 2.0
    xfeat_integration_mode: str = "assist"
    xfeat_assist_min_legacy_points: int = 0
    xfeat_repo_dir: Optional[Path] = None
    xfeat_git_ref: str = DEFAULT_ACCELERATED_FEATURES_REF
    xfeat_auto_bootstrap: bool = True


@dataclass
class MatchResult:
    prev_points: np.ndarray
    cur_points: np.ndarray
    force_redetect: bool = False
    scores: Optional[np.ndarray] = None
    backend: str = ""
    raw_matches_total: int = 0
    ransac_threshold: Optional[float] = None


class BaseMatcher:
    name = "base"
    requires_seed_points = True

    def __init__(self, config: MarkerMatcherConfig):
        self.config = config

    def detect_features(self, gray: np.ndarray, mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
        raise NotImplementedError

    def match(
        self,
        prev_frame_bgr: np.ndarray,
        cur_frame_bgr: np.ndarray,
        prev_gray: np.ndarray,
        cur_gray: np.ndarray,
        prev_roi: Optional[np.ndarray],
        cur_roi: Optional[np.ndarray],
        prev_points: Optional[np.ndarray],
        h_guess_prev_to_cur: Optional[np.ndarray] = None,
    ) -> MatchResult:
        raise NotImplementedError


class LegacyMatcher(BaseMatcher):
    name = NAV_MATCHER_LEGACY
    requires_seed_points = True

    def detect_features(self, gray: np.ndarray, mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
        return cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.config.max_corners,
            qualityLevel=self.config.quality_level,
            minDistance=self.config.min_distance,
            blockSize=self.config.block_size,
            mask=mask,
        )

    def match(
        self,
        prev_frame_bgr: np.ndarray,
        cur_frame_bgr: np.ndarray,
        prev_gray: np.ndarray,
        cur_gray: np.ndarray,
        prev_roi: Optional[np.ndarray],
        cur_roi: Optional[np.ndarray],
        prev_points: Optional[np.ndarray],
        h_guess_prev_to_cur: Optional[np.ndarray] = None,
    ) -> MatchResult:
        _ = (prev_frame_bgr, cur_frame_bgr, prev_roi, cur_roi)
        empty = np.empty((0, 2), dtype=np.float32)
        if prev_points is None or len(prev_points) < self.config.min_track_points:
            return MatchResult(empty, empty, backend=self.name)

        lk_win = self.config.lk_win if (self.config.lk_win % 2 == 1) else (self.config.lk_win + 1)
        lk_params = dict(
            winSize=(lk_win, lk_win),
            maxLevel=self.config.lk_levels,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        lk_flags = 0
        next_init = None
        if h_guess_prev_to_cur is not None:
            try:
                next_init = cv2.perspectiveTransform(prev_points, h_guess_prev_to_cur)
                lk_flags |= cv2.OPTFLOW_USE_INITIAL_FLOW
            except cv2.error:
                next_init = None
                lk_flags = 0

        cur_points, st_fwd, err_fwd = cv2.calcOpticalFlowPyrLK(
            prev_gray, cur_gray, prev_points, next_init, flags=lk_flags, **lk_params
        )
        if cur_points is None or st_fwd is None:
            return MatchResult(empty, empty, backend=self.name)

        st_fwd = st_fwd.reshape(-1).astype(bool)
        if np.count_nonzero(st_fwd) < self.config.min_track_points:
            return MatchResult(empty, empty, backend=self.name)

        p0_f = prev_points[st_fwd]
        p1_f = cur_points[st_fwd]
        good = np.ones((len(p0_f),), dtype=bool)

        if err_fwd is not None and self.config.lk_error_threshold > 0:
            ef = err_fwd.reshape(-1)[st_fwd]
            good &= np.isfinite(ef) & (ef < self.config.lk_error_threshold)

        if self.config.fb_threshold > 0:
            back_points, st_back, err_back = cv2.calcOpticalFlowPyrLK(cur_gray, prev_gray, p1_f, None, **lk_params)
            if back_points is not None and st_back is not None:
                st_back = st_back.reshape(-1).astype(bool)
                good &= st_back

                fb = np.linalg.norm(p0_f.reshape(-1, 2) - back_points.reshape(-1, 2), axis=1)
                good &= np.isfinite(fb) & (fb < self.config.fb_threshold)

                if err_back is not None and self.config.lk_error_threshold > 0:
                    eb = err_back.reshape(-1)
                    good &= np.isfinite(eb) & (eb < self.config.lk_error_threshold)

        p0 = p0_f.reshape(-1, 2)[good].astype(np.float32)
        p1 = p1_f.reshape(-1, 2)[good].astype(np.float32)
        force_redetect = self.config.redetect_min_points > 0 and len(p0) < self.config.redetect_min_points
        return MatchResult(p0, p1, force_redetect=force_redetect, backend=self.name, raw_matches_total=len(p0))


class XFeatLighterGlueMatcher(BaseMatcher):
    name = NAV_MATCHER_XFEAT_LIGHTERGLUE
    requires_seed_points = False

    def __init__(self, config: MarkerMatcherConfig, initialize: bool = True):
        super().__init__(config)
        self._model = None
        self._torch = None
        self._legacy = LegacyMatcher(config)
        if initialize:
            self._ensure_model()

    def detect_features(self, gray: np.ndarray, mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
        return self._legacy.detect_features(gray, mask)

    def match(
        self,
        prev_frame_bgr: np.ndarray,
        cur_frame_bgr: np.ndarray,
        prev_gray: np.ndarray,
        cur_gray: np.ndarray,
        prev_roi: Optional[np.ndarray],
        cur_roi: Optional[np.ndarray],
        prev_points: Optional[np.ndarray],
        h_guess_prev_to_cur: Optional[np.ndarray] = None,
    ) -> MatchResult:
        mode = self.config.xfeat_integration_mode.strip().lower()
        if mode == "assist":
            legacy_result = self._legacy.match(
                prev_frame_bgr,
                cur_frame_bgr,
                prev_gray,
                cur_gray,
                prev_roi,
                cur_roi,
                prev_points,
                h_guess_prev_to_cur,
            )
            min_legacy = self.config.xfeat_assist_min_legacy_points or self.config.min_track_points
            if len(legacy_result.prev_points) >= min_legacy:
                legacy_result.backend = "legacy_assist"
                return legacy_result

            xfeat_result = self.match_xfeat(
                prev_frame_bgr,
                cur_frame_bgr,
                prev_roi,
                cur_roi,
                h_guess_prev_to_cur,
            )
            xfeat_result.backend = "xfeat_recovery"
            return xfeat_result

        return self.match_xfeat(prev_frame_bgr, cur_frame_bgr, prev_roi, cur_roi, h_guess_prev_to_cur)

    def match_xfeat(
        self,
        prev_frame_bgr: np.ndarray,
        cur_frame_bgr: np.ndarray,
        prev_roi: Optional[np.ndarray],
        cur_roi: Optional[np.ndarray],
        h_guess_prev_to_cur: Optional[np.ndarray] = None,
    ) -> MatchResult:
        self._ensure_model()
        empty = np.empty((0, 2), dtype=np.float32)

        d0 = self._detect_and_filter(prev_frame_bgr, prev_roi)
        d1 = self._detect_and_filter(cur_frame_bgr, cur_roi)
        if len(d0["keypoints"]) < 4 or len(d1["keypoints"]) < 4:
            return MatchResult(empty, empty, force_redetect=True, backend=self.name)

        p0, p1, scores = self._match_lighterglue_with_scores(d0, d1)
        raw_total = len(p0)
        p0, p1, scores = self._post_filter_matches(p0, p1, scores, prev_roi, cur_roi, h_guess_prev_to_cur)
        return MatchResult(
            p0,
            p1,
            force_redetect=len(p0) < self.config.min_track_points,
            scores=scores,
            backend=self.name,
            raw_matches_total=raw_total,
            ransac_threshold=self.config.xfeat_ransac_threshold,
        )

    def _match_lighterglue_with_scores(self, d0, d1) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        torch = self._torch
        if not self._model.kornia_available:
            raise RuntimeError("We rely on kornia for LightGlue. Install with: pip install kornia")
        if self._model.lighterglue is None:
            from modules.lighterglue import LighterGlue

            self._model.lighterglue = LighterGlue()

        data = {
            "keypoints0": d0["keypoints"][None, ...],
            "keypoints1": d1["keypoints"][None, ...],
            "descriptors0": d0["descriptors"][None, ...],
            "descriptors1": d1["descriptors"][None, ...],
            "image_size0": torch.tensor(d0["image_size"]).to(self._model.dev)[None, ...],
            "image_size1": torch.tensor(d1["image_size"]).to(self._model.dev)[None, ...],
        }
        out = self._model.lighterglue(data, min_conf=self.config.xfeat_min_conf)
        idxs = out["matches"][0]
        if len(idxs) == 0:
            empty = np.empty((0, 2), dtype=np.float32)
            return empty, empty, np.empty((0,), dtype=np.float32)

        p0 = d0["keypoints"][idxs[:, 0]].detach().cpu().numpy().astype(np.float32)
        p1 = d1["keypoints"][idxs[:, 1]].detach().cpu().numpy().astype(np.float32)
        scores = None
        raw_scores = out.get("scores")
        if raw_scores is not None and len(raw_scores) > 0:
            scores = raw_scores[0].detach().cpu().numpy().astype(np.float32)
        if scores is None or len(scores) != len(p0):
            s0 = d0["scores"][idxs[:, 0]]
            s1 = d1["scores"][idxs[:, 1]]
            scores = torch.sqrt(torch.clamp(s0 * s1, min=0)).detach().cpu().numpy().astype(np.float32)
        return p0.reshape(-1, 2), p1.reshape(-1, 2), scores.reshape(-1)

    def _post_filter_matches(
        self,
        p0: np.ndarray,
        p1: np.ndarray,
        scores: Optional[np.ndarray],
        prev_roi: Optional[np.ndarray],
        cur_roi: Optional[np.ndarray],
        h_guess_prev_to_cur: Optional[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        if len(p0) == 0:
            return p0, p1, scores

        keep = np.ones((len(p0),), dtype=bool)
        keep &= _points_inside_mask(p0, prev_roi)
        keep &= _points_inside_mask(p1, cur_roi)
        p0 = p0[keep]
        p1 = p1[keep]
        scores = scores[keep] if scores is not None and len(scores) == len(keep) else scores

        if scores is not None and len(scores) == len(p0):
            order = np.argsort(-scores)
            max_matches = max(4, int(self.config.xfeat_max_matches))
            order = order[:max_matches]
            p0 = p0[order]
            p1 = p1[order]
            scores = scores[order]

        if len(p0) < self.config.min_track_points:
            return p0.astype(np.float32), p1.astype(np.float32), scores

        motion_keep = self._motion_gate(p0, p1, h_guess_prev_to_cur)
        if np.count_nonzero(motion_keep) >= self.config.min_track_points:
            p0 = p0[motion_keep]
            p1 = p1[motion_keep]
            scores = scores[motion_keep] if scores is not None and len(scores) == len(motion_keep) else scores

        return p0.astype(np.float32), p1.astype(np.float32), scores

    def _motion_gate(
        self,
        p0: np.ndarray,
        p1: np.ndarray,
        h_guess_prev_to_cur: Optional[np.ndarray],
    ) -> np.ndarray:
        limit = float(self.config.xfeat_motion_gate_px)
        if limit <= 0:
            return np.ones((len(p0),), dtype=bool)

        if h_guess_prev_to_cur is not None:
            try:
                pred = cv2.perspectiveTransform(p0.reshape(-1, 1, 2).astype(np.float32), h_guess_prev_to_cur).reshape(-1, 2)
                err = np.linalg.norm(pred - p1, axis=1)
                keep = np.isfinite(err) & (err <= limit)
                if np.count_nonzero(keep) >= self.config.min_track_points:
                    return keep
            except cv2.error:
                pass

        delta = p1 - p0
        med = np.median(delta, axis=0)
        dev = np.linalg.norm(delta - med, axis=1)
        mad = float(np.median(np.abs(dev - np.median(dev)))) if len(dev) else 0.0
        adaptive_limit = max(limit, 3.5 * 1.4826 * mad)
        return np.isfinite(dev) & (dev <= adaptive_limit)

    def _detect_and_filter(self, frame_bgr: np.ndarray, roi_mask: Optional[np.ndarray]):
        h, w = frame_bgr.shape[:2]
        image = self._torch.from_numpy(np.ascontiguousarray(frame_bgr)).permute(2, 0, 1)[None].float() / 255.0
        out = self._model.detectAndCompute(image, top_k=self.config.xfeat_top_k)[0]
        # XFeat rescales keypoints back to this tensor's input size, which is the marker pipeline frame size.
        out["image_size"] = (w, h)
        if roi_mask is None or len(out["keypoints"]) == 0:
            return out

        torch = self._torch
        kpts = out["keypoints"]
        xs = torch.clamp(torch.round(kpts[:, 0]).long(), 0, w - 1)
        ys = torch.clamp(torch.round(kpts[:, 1]).long(), 0, h - 1)
        roi_tensor = torch.as_tensor(roi_mask, device=kpts.device)
        keep = roi_tensor[ys, xs] > 0
        out["keypoints"] = out["keypoints"][keep]
        out["scores"] = out["scores"][keep]
        out["descriptors"] = out["descriptors"][keep]
        return out

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
        except Exception as exc:  # pragma: no cover - dependency error path
            raise RuntimeError("NAV_MATCHER=xfeat_lighterglue requires torch to be installed.") from exc

        self._torch = torch
        repo_dir = self._prepare_accelerated_features_repo()
        if repo_dir is not None:
            self._load_from_repo(repo_dir)
            return

        try:
            self._model = torch.hub.load(
                ACCELERATED_FEATURES_HUB,
                "XFeat",
                pretrained=True,
                top_k=self.config.xfeat_top_k,
                detection_threshold=self.config.xfeat_detection_threshold,
            )
        except Exception as exc:  # pragma: no cover - network/runtime dependent
            raise RuntimeError(
                "Failed to load official verlab/accelerated_features through torch.hub. "
                "Run scripts/bootstrap_accelerated_features.py or set XFEAT_REPO_DIR to a local clone."
            ) from exc

        try:
            from modules.lighterglue import LighterGlue

            self._model.lighterglue = LighterGlue()
        except Exception as exc:  # pragma: no cover - torch.hub internals are runtime dependent
            raise RuntimeError(
                "Loaded XFeat, but failed to initialize the official LighterGlue module. "
                "Install kornia==0.7.2 and use a local accelerated_features clone if torch.hub did not expose modules."
            ) from exc

    def _load_from_repo(self, repo_dir: Path) -> None:
        repo_str = str(repo_dir.resolve())
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

        try:
            from modules.lighterglue import LighterGlue
            from modules.xfeat import XFeat
        except Exception as exc:  # pragma: no cover - external repo import path
            raise RuntimeError(f"Failed to import official XFeat modules from {repo_dir}") from exc

        xfeat_weights = repo_dir / "weights" / "xfeat.pt"
        if _looks_like_real_weight_file(xfeat_weights):
            weights = str(xfeat_weights)
        else:
            weights = self._torch.hub.load_state_dict_from_url(XFEAT_WEIGHTS_URL)
        self._model = XFeat(
            weights=weights,
            top_k=self.config.xfeat_top_k,
            detection_threshold=self.config.xfeat_detection_threshold,
        )

        lighterglue_weights = repo_dir / "weights" / "xfeat-lighterglue.pt"
        if _looks_like_real_weight_file(lighterglue_weights):
            self._model.lighterglue = LighterGlue(weights=str(lighterglue_weights))
        else:
            self._model.lighterglue = LighterGlue(weights=str(repo_dir / "weights" / "__download_xfeat-lighterglue.pt"))

    def _prepare_accelerated_features_repo(self) -> Optional[Path]:
        repo_dir = self.config.xfeat_repo_dir or DEFAULT_ACCELERATED_FEATURES_DIR
        if _has_official_xfeat_modules(repo_dir):
            return repo_dir

        if repo_dir.exists() and not _has_official_xfeat_modules(repo_dir):
            raise RuntimeError(f"XFEAT_REPO_DIR exists but does not look like accelerated_features: {repo_dir}")

        if not self.config.xfeat_auto_bootstrap:
            return None

        git_bin = shutil.which("git")
        if git_bin is None:
            return None

        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        _run_checked([git_bin, "clone", "--depth", "1", ACCELERATED_FEATURES_URL, str(repo_dir)])
        ref = (self.config.xfeat_git_ref or "").strip()
        if ref and ref not in {"main", "master"}:
            _run_checked([git_bin, "-C", str(repo_dir), "fetch", "--depth", "1", "origin", ref])
            _run_checked([git_bin, "-C", str(repo_dir), "checkout", ref])
        return repo_dir if _has_official_xfeat_modules(repo_dir) else None


def normalize_nav_matcher_name(name: str | None) -> str:
    value = (name or NAV_MATCHER_LEGACY).strip().lower().replace("-", "_")
    aliases = {
        "": NAV_MATCHER_LEGACY,
        "lk": NAV_MATCHER_LEGACY,
        "opencv": NAV_MATCHER_LEGACY,
        "xfeat": NAV_MATCHER_XFEAT_LIGHTERGLUE,
        "xfeat_lightglue": NAV_MATCHER_XFEAT_LIGHTERGLUE,
        "xfeat_lg": NAV_MATCHER_XFEAT_LIGHTERGLUE,
    }
    value = aliases.get(value, value)
    if value not in VALID_NAV_MATCHERS:
        allowed = ", ".join(sorted(VALID_NAV_MATCHERS))
        raise ValueError(f"Unknown NAV_MATCHER={name!r}. Expected one of: {allowed}")
    return value


def marker_matcher_config_from_env(**params) -> MarkerMatcherConfig:
    return MarkerMatcherConfig(
        **params,
        xfeat_top_k=max(4, int(os.getenv("XFEAT_TOP_K", "800"))),
        xfeat_detection_threshold=float(os.getenv("XFEAT_DETECTION_THRESHOLD", "0.05")),
        xfeat_min_conf=float(os.getenv("XFEAT_LIGHTERGLUE_MIN_CONF", "0.25")),
        xfeat_max_matches=max(4, int(os.getenv("XFEAT_MAX_MATCHES", "300"))),
        xfeat_motion_gate_px=float(os.getenv("XFEAT_MOTION_GATE_PX", "35.0")),
        xfeat_ransac_threshold=float(os.getenv("XFEAT_RANSAC_THR", "2.0")),
        xfeat_integration_mode=os.getenv("XFEAT_INTEGRATION_MODE", "assist").strip().lower() or "assist",
        xfeat_assist_min_legacy_points=max(0, int(os.getenv("XFEAT_ASSIST_MIN_LEGACY_PTS", "0"))),
        xfeat_repo_dir=_env_path("XFEAT_REPO_DIR"),
        xfeat_git_ref=os.getenv("XFEAT_GIT_REF", DEFAULT_ACCELERATED_FEATURES_REF).strip() or DEFAULT_ACCELERATED_FEATURES_REF,
        xfeat_auto_bootstrap=_env_bool("XFEAT_AUTO_BOOTSTRAP", True),
    )


def build_marker_matcher(name: str | None, config: MarkerMatcherConfig, initialize: bool = True) -> BaseMatcher:
    normalized = normalize_nav_matcher_name(name)
    if normalized == NAV_MATCHER_LEGACY:
        return LegacyMatcher(config)
    if normalized == NAV_MATCHER_XFEAT_LIGHTERGLUE:
        return XFeatLighterGlueMatcher(config, initialize=initialize)
    raise AssertionError(f"Unhandled matcher backend: {normalized}")


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_path(key: str) -> Optional[Path]:
    raw = os.getenv(key, "").strip()
    return Path(raw) if raw else None


def _has_official_xfeat_modules(path: Path) -> bool:
    return (path / "modules" / "xfeat.py").exists() and (path / "modules" / "lighterglue.py").exists()


def _looks_like_real_weight_file(path: Path) -> bool:
    if not path.exists() or not path.is_file() or path.stat().st_size < 4096:
        return False
    try:
        with path.open("rb") as fh:
            head = fh.read(64)
    except OSError:
        return False
    return not head.startswith(b"version https://git-lfs")


def _points_inside_mask(points: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    if mask is None:
        return np.ones((len(points),), dtype=bool)
    if len(points) == 0:
        return np.zeros((0,), dtype=bool)
    h, w = mask.shape[:2]
    xs = np.clip(np.round(points[:, 0]).astype(np.int32), 0, max(0, w - 1))
    ys = np.clip(np.round(points[:, 1]).astype(np.int32), 0, max(0, h - 1))
    return mask[ys, xs] > 0


def _run_checked(cmd: list[str]) -> None:
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as exc:  # pragma: no cover - network/runtime dependent
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        raise RuntimeError(f"Command failed while preparing accelerated_features: {' '.join(cmd)}\n{detail}") from exc
