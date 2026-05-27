"""Tracking module — pure-logic tests (no GPU, no heavy weights)."""

from __future__ import annotations

import numpy as np

from mvmm.tracking.analytics import DwellTimer, LineCounter, PolygonZone
from mvmm.tracking.byte_track import Track
from mvmm.tracking.detectors import Detections


def _trk(tid: int, xy: tuple[float, float], cls: int = 0, cls_name: str = "person") -> Track:
    x, y = xy
    return Track(
        track_id=tid,
        bbox=np.array([x - 5, y - 10, x + 5, y]),
        score=0.9,
        class_id=cls,
        class_name=cls_name,
    )


def test_detections_filter_by_score():
    d = Detections(
        boxes=np.array([[0, 0, 10, 10], [10, 10, 20, 20]], dtype=np.float32),
        scores=np.array([0.1, 0.9], dtype=np.float32),
        labels=np.array([0, 0]),
        class_names=["person"],
    )
    f = d.filter_by_score(0.5)
    assert len(f.boxes) == 1 and f.scores[0] > 0.5


def test_detections_filter_by_class():
    d = Detections(
        boxes=np.array([[0, 0, 10, 10], [10, 10, 20, 20]], dtype=np.float32),
        scores=np.array([0.9, 0.9], dtype=np.float32),
        labels=np.array([0, 1]),
        class_names=["person", "forklift"],
    )
    f = d.filter_by_class(["forklift"])
    assert len(f.boxes) == 1 and f.labels[0] == 1


def test_polygon_zone_count_inside():
    poly = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
    zone = PolygonZone(polygon=poly, name="floor")
    inside = _trk(1, (50, 50))
    outside = _trk(2, (200, 200))
    assert zone.count([inside, outside]) == 1


def test_line_counter_directional_crossing():
    lc = LineCounter(a=(0, 50), b=(100, 50), name="entry")
    t = _trk(1, (50, 20))  # above the line (cross > 0 with our convention)
    lc.update([t])
    t.bbox = np.array([45, 50, 55, 80])  # foot now below the line
    lc.update([t])
    assert (lc.in_count + lc.out_count) == 1


def test_dwell_timer_accumulates():
    zone = PolygonZone(polygon=np.array([[0, 0], [100, 0], [100, 100], [0, 100]]), name="z")
    dwell = DwellTimer(zone=zone)
    t = _trk(1, (50, 50))
    for _ in range(30):
        dwell.update([t])
    assert dwell.dwell_frames[1] == 30
    secs = dwell.seconds(fps=30.0)
    assert secs[1] == 1.0
