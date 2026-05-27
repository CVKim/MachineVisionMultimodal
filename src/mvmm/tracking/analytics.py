"""Tracking analytics — zone counting, line crossing, dwell time.

These are the small but absolutely critical post-processing modules
that turn a tracker's bounding boxes into actual business metrics:

    * `PolygonZone.count(tracks)`  — how many objects inside a zone
    * `LineCounter.update(tracks)` — line-crossing count (entry/exit)
    * `DwellTimer.update(tracks)`  — per-track time inside a zone
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from mvmm.tracking.byte_track import Track


def _bbox_center(b: np.ndarray) -> tuple[float, float]:
    return float((b[0] + b[2]) / 2.0), float((b[1] + b[3]) / 2.0)


def _bbox_foot(b: np.ndarray) -> tuple[float, float]:
    """Approx ground-contact point: bottom-center of the bbox."""
    return float((b[0] + b[2]) / 2.0), float(b[3])


def _point_in_polygon(p: tuple[float, float], poly: np.ndarray) -> bool:
    """Ray-casting point-in-polygon. ``poly`` is (N,2)."""
    x, y = p
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        intersect = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
        if intersect:
            inside = not inside
        j = i
    return inside


@dataclass
class PolygonZone:
    """Counts how many tracks have their foot inside a polygon."""

    polygon: np.ndarray  # (N, 2)
    name: str = "zone"

    def contains(self, track: Track) -> bool:
        return _point_in_polygon(_bbox_foot(track.bbox), self.polygon)

    def count(self, tracks: list[Track]) -> int:
        return sum(1 for t in tracks if self.contains(t))


@dataclass
class LineCounter:
    """Count how many tracks cross a directed line (A -> B).

    Direction matters: crossing from the *left* side of A->B to the right
    counts as +1 (entry); right-to-left is -1 (exit). Each track is
    counted at most once per direction (latching on transition).
    """

    a: tuple[float, float]
    b: tuple[float, float]
    name: str = "line"
    _side: dict[int, int] = field(default_factory=dict)
    in_count: int = 0
    out_count: int = 0

    def _side_of(self, p: tuple[float, float]) -> int:
        ax, ay = self.a
        bx, by = self.b
        cross = (bx - ax) * (p[1] - ay) - (by - ay) * (p[0] - ax)
        return 1 if cross > 0 else (-1 if cross < 0 else 0)

    def update(self, tracks: list[Track]) -> None:
        for t in tracks:
            side = self._side_of(_bbox_foot(t.bbox))
            prev = self._side.get(t.track_id)
            if prev is not None and side != 0 and prev != 0 and prev != side:
                if side > prev:
                    self.in_count += 1
                else:
                    self.out_count += 1
            if side != 0:
                self._side[t.track_id] = side


@dataclass
class DwellTimer:
    """Per-track dwell time (in frames) inside a polygon zone."""

    zone: PolygonZone
    dwell_frames: dict[int, int] = field(default_factory=lambda: defaultdict(int))

    def update(self, tracks: list[Track]) -> None:
        for t in tracks:
            if self.zone.contains(t):
                self.dwell_frames[t.track_id] += 1

    def seconds(self, fps: float) -> dict[int, float]:
        return {k: v / fps for k, v in self.dwell_frames.items()}
