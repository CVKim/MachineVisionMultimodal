"""Hazard detector tests — no model deps required."""

from __future__ import annotations

import numpy as np

from mvmm.tracking.analytics import PolygonZone
from mvmm.tracking.hazard import HazardDetector, _bbox_distance_px
from mvmm.tracking.pose_activity import PoseTrack


def _trk(tid: int, xyxy: tuple[float, float, float, float]) -> PoseTrack:
    return PoseTrack(
        track_id=tid,
        bbox=np.array(xyxy, dtype=np.float32),
        keypoints=np.zeros((17, 2), dtype=np.float32),
        kp_conf=np.zeros(17, dtype=np.float32),
        score=0.9,
    )


def test_zone_violation_fires_once_then_cooldown():
    zone = PolygonZone(polygon=np.array([[0, 0], [100, 0], [100, 100], [0, 100]]), name="floor")
    det = HazardDetector(restricted_zones=[zone], cool_down_s=1.0)
    # Track inside the zone for 3 successive frames at the same timestamp window.
    inside = _trk(1, (40, 40, 60, 80))
    e0 = det.step([inside], frame_rgb=None, timestamp=0.0)
    e1 = det.step([inside], frame_rgb=None, timestamp=0.1)
    assert len(e0) == 1 and e0[0].type == "zone_violation"
    # The detector should *not* re-fire within the cooldown window.
    assert e1 == []


def test_zone_violation_skipped_when_outside():
    zone = PolygonZone(polygon=np.array([[0, 0], [100, 0], [100, 100], [0, 100]]), name="floor")
    det = HazardDetector(restricted_zones=[zone])
    outside = _trk(1, (200, 200, 240, 280))
    assert det.step([outside], frame_rgb=None, timestamp=0.0) == []


def test_proximity_fires_when_close():
    det = HazardDetector(proximity_threshold_px=20.0)
    a = _trk(1, (0, 0, 10, 10))
    b = _trk(2, (12, 0, 22, 10))  # foot dist ~17 px
    events = det.step([a, b], frame_rgb=None, timestamp=0.0)
    assert len(events) == 1 and events[0].type == "proximity"


def test_proximity_skipped_when_far():
    det = HazardDetector(proximity_threshold_px=20.0)
    a = _trk(1, (0, 0, 10, 10))
    b = _trk(2, (100, 0, 110, 10))
    assert det.step([a, b], frame_rgb=None, timestamp=0.0) == []


def test_bbox_distance_helper():
    a = np.array([0, 0, 10, 10], dtype=np.float32)
    b = np.array([20, 0, 30, 10], dtype=np.float32)
    assert _bbox_distance_px(a, b) == 20.0


def test_summary_counts_by_type():
    zone = PolygonZone(polygon=np.array([[0, 0], [100, 0], [100, 100], [0, 100]]), name="z")
    det = HazardDetector(restricted_zones=[zone], proximity_threshold_px=20.0, cool_down_s=10.0)
    a, b = _trk(1, (20, 20, 40, 60)), _trk(2, (30, 20, 50, 60))
    det.step([a, b], frame_rgb=None, timestamp=0.0)
    s = det.summary()
    assert s["n_events"] >= 1
    assert "zone_violation" in s["by_type"] or "proximity" in s["by_type"]
