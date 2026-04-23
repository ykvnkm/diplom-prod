from __future__ import annotations

import os
import unittest
from pathlib import Path

import cv2
import numpy as np

from services.unified_runtime.navigation_matchers import (
    LegacyMatcher,
    MarkerMatcherConfig,
    XFeatLighterGlueMatcher,
    build_marker_matcher,
    normalize_nav_matcher_name,
)


def matcher_config(min_track_points: int = 8) -> MarkerMatcherConfig:
    return MarkerMatcherConfig(
        max_corners=200,
        min_track_points=min_track_points,
        quality_level=0.01,
        min_distance=5,
        block_size=7,
        lk_win=21,
        lk_levels=3,
        fb_threshold=1.5,
        lk_error_threshold=25.0,
        redetect_min_points=0,
        xfeat_top_k=128,
    )


class MatcherFactoryTests(unittest.TestCase):
    def test_factory_selects_legacy_by_default(self) -> None:
        matcher = build_marker_matcher("legacy", matcher_config())
        self.assertIsInstance(matcher, LegacyMatcher)
        self.assertEqual(normalize_nav_matcher_name("lk"), "legacy")

    def test_factory_selects_xfeat_without_eager_model_load(self) -> None:
        matcher = build_marker_matcher("xfeat_lighterglue", matcher_config(), initialize=False)
        self.assertIsInstance(matcher, XFeatLighterGlueMatcher)
        self.assertEqual(normalize_nav_matcher_name("xfeat_lightglue"), "xfeat_lighterglue")


class LegacyPipelineSmokeTests(unittest.TestCase):
    def test_marker_pipeline_runs_with_legacy_matcher(self) -> None:
        from services.unified_runtime import unified_navigation_service as service

        class FakeReader:
            def __init__(self, frames: list[np.ndarray]):
                self.frames = frames
                self.idx = 0

            def is_open(self) -> bool:
                return True

            def fps(self) -> float:
                return 5.0

            def read(self):
                if self.idx >= len(self.frames):
                    return False, None
                frame = self.frames[self.idx]
                self.idx += 1
                return True, frame.copy()

            def close(self) -> None:
                pass

        frame = np.zeros((540, 960, 3), dtype=np.uint8)
        cv2.rectangle(frame, (340, 250), (620, 510), (0, 0, 255), -1)
        cv2.line(frame, (340, 250), (620, 510), (255, 255, 255), 2)
        cv2.line(frame, (620, 250), (340, 510), (255, 255, 255), 2)
        updates: list[dict] = []
        service.REPORT_DIR = Path("tmp/test_reports")
        service.REPORT_DIR.mkdir(parents=True, exist_ok=True)

        service.run_unified_pipeline(
            "test_legacy_pipeline",
            FakeReader([frame, frame]),
            None,
            updates.append,
            service.SourceProfile(detection_stride=1, emit_only_detections=False),
            "marker",
            False,
            "test_legacy_pipeline",
            "yolo",
            None,
            None,
            None,
            None,
            None,
            None,
            "test:local:video",
            nav_matcher="legacy",
        )

        self.assertTrue(any(msg.get("info") == "Navigation matcher: legacy" for msg in updates))
        self.assertTrue(updates[-1].get("done"))
        self.assertEqual(updates[-1].get("nav_matcher"), "legacy")


class XFeatGuardedSmokeTests(unittest.TestCase):
    @unittest.skipUnless(os.getenv("RUN_XFEAT_SMOKE") == "1", "set RUN_XFEAT_SMOKE=1 to run the heavy XFeat smoke test")
    def test_xfeat_lighterglue_matcher_can_run_one_match(self) -> None:
        cfg = matcher_config(min_track_points=4)
        matcher = XFeatLighterGlueMatcher(cfg, initialize=True)

        img = np.zeros((160, 160, 3), dtype=np.uint8)
        for i in range(20, 145, 20):
            cv2.circle(img, (i, 40 + (i % 60)), 4, (255, 255, 255), -1)
        cv2.putText(img, "X", (65, 105), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        result = matcher.match(img, img, gray, gray, None, None, None)
        self.assertGreaterEqual(len(result.prev_points), 4)
        self.assertEqual(result.prev_points.shape, result.cur_points.shape)


if __name__ == "__main__":
    unittest.main()
