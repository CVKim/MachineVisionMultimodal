"""Safety-hazard detection layer on top of tracking.

Three hazard types are supported out of the box:

    * **Zone violation** — a tracked person enters a user-defined
      restricted polygon (e.g. "forklift lane", "press machine
      perimeter"). Uses :class:`mvmm.tracking.analytics.PolygonZone`.
    * **Proximity** — two tracks come closer than a pixel threshold
      (proxy for unsafe-distance violations between worker pairs or
      worker/forklift).
    * **PPE missing** — GroundingDINO is run on each person crop with
      prompts like "person wearing safety helmet" / "person without
      hard hat". A track that fails the required-PPE checks fires a
      ``ppe_missing`` event.

Every event carries a ``severity`` ∈ {"warn", "danger"} and a
machine-readable detail dict. Events are emitted *only on transitions*
(first frame the condition holds for a given track) so the consumer
doesn't drown in duplicates — internal state tracks who is already
"flagged" and clears when they leave the condition for ``cool_down_s``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from math import inf
from typing import Any

import numpy as np

from mvmm.tracking.analytics import PolygonZone
from mvmm.tracking.pose_activity import PoseTrack


@dataclass
class HazardEvent:
    """One detected safety hazard."""

    timestamp: float
    track_id: int
    type: str  # "zone_violation" | "proximity" | "ppe_missing"
    severity: str  # "warn" | "danger"
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "timestamp": round(self.timestamp, 4),
            "track_id": self.track_id,
            "type": self.type,
            "severity": self.severity,
            "details": self.details,
        }


def _bbox_foot(b: np.ndarray) -> tuple[float, float]:
    """Approximate ground-contact point: bottom-center of the bbox."""
    return float((b[0] + b[2]) / 2.0), float(b[3])


def _bbox_distance_px(a: np.ndarray, b: np.ndarray) -> float:
    """Approximate distance between two bboxes via their foot points."""
    fa, fb = _bbox_foot(a), _bbox_foot(b)
    return float(np.hypot(fa[0] - fb[0], fa[1] - fb[1]))


# ---------------------------------------------------------------------------
class HazardDetector:
    """Compose zone / proximity / PPE checks on every frame.

    Args:
        restricted_zones:        list of PolygonZone instances treated as
                                 "no-go" areas. Anyone whose foot is
                                 inside triggers a ``zone_violation``.
        proximity_threshold_px:  pixel distance below which two tracks
                                 trigger a ``proximity`` warning. Set to
                                 None to disable.
        ppe_detector:            optional GroundingDINODetector. When
                                 set, the per-track crop is sent through
                                 the detector every ``ppe_every_n``
                                 frames with ``ppe_prompts``; absent
                                 boxes fire ``ppe_missing``.
        ppe_prompts:             list of free-text PPE classes — e.g.
                                 ``["safety helmet", "hard hat",
                                 "high-visibility vest"]``.
        ppe_every_n:             only run the PPE detector every Nth
                                 frame per track (it's slow).
        cool_down_s:             once a track *exits* a hazard
                                 condition, wait this long before
                                 firing again. Default 5 seconds.
    """

    def __init__(
        self,
        restricted_zones: list[PolygonZone] | None = None,
        proximity_threshold_px: float | None = None,
        ppe_detector: Any | None = None,
        ppe_prompts: list[str] | None = None,
        ppe_every_n: int = 30,
        cool_down_s: float = 5.0,
    ):
        self.zones = list(restricted_zones or [])
        self.proximity_threshold_px = (
            None if proximity_threshold_px is None else float(proximity_threshold_px)
        )
        self.ppe_detector = ppe_detector
        self.ppe_prompts = list(ppe_prompts or [])
        self.ppe_every_n = max(int(ppe_every_n), 1)
        self.cool_down_s = float(cool_down_s)
        # Per-track / per-zone active state — last frame the condition was true.
        self._active_zone: dict[tuple[int, str], float] = defaultdict(float)
        self._active_proximity: dict[tuple[int, int], float] = defaultdict(float)
        self._active_ppe: dict[int, float] = defaultdict(float)
        self._frames_seen: int = 0
        self._all_events: list[HazardEvent] = []

    # ----------------------------------------------------------------- step
    def step(
        self,
        tracks: list[PoseTrack],
        frame_rgb: np.ndarray | None,
        timestamp: float,
    ) -> list[HazardEvent]:
        """Run all configured hazard checks for one frame.

        Returns the list of *new* events fired this frame.
        """
        self._frames_seen += 1
        events: list[HazardEvent] = []

        # 1) Zone violations.
        for t in tracks:
            for z in self.zones:
                key = (t.track_id, z.name)
                if z.contains(t):
                    last = self._active_zone.get(key, -inf)
                    if (timestamp - last) > self.cool_down_s:
                        events.append(
                            HazardEvent(
                                timestamp=timestamp,
                                track_id=t.track_id,
                                type="zone_violation",
                                severity="danger",
                                details={"zone": z.name},
                            )
                        )
                    self._active_zone[key] = timestamp

        # 2) Pairwise proximity.
        if self.proximity_threshold_px is not None:
            for i, ta in enumerate(tracks):
                for tb in tracks[i + 1 :]:
                    d = _bbox_distance_px(ta.bbox, tb.bbox)
                    if d > self.proximity_threshold_px:
                        continue
                    key = tuple(sorted((ta.track_id, tb.track_id)))
                    last = self._active_proximity.get(key, -inf)
                    if (timestamp - last) > self.cool_down_s:
                        events.append(
                            HazardEvent(
                                timestamp=timestamp,
                                track_id=ta.track_id,
                                type="proximity",
                                severity="warn",
                                details={
                                    "other_track_id": tb.track_id,
                                    "distance_px": round(d, 1),
                                    "threshold_px": self.proximity_threshold_px,
                                },
                            )
                        )
                    self._active_proximity[key] = timestamp

        # 3) PPE compliance — slow, optional, throttled per track.
        if (
            self.ppe_detector is not None
            and self.ppe_prompts
            and frame_rgb is not None
            and self._frames_seen % self.ppe_every_n == 0
        ):
            for t in tracks:
                crop = self._crop(frame_rgb, t.bbox)
                if crop is None or crop.size == 0:
                    continue
                try:
                    dets = self.ppe_detector(crop, classes=self.ppe_prompts)
                except Exception:
                    continue
                missing = self._missing_ppe(dets, self.ppe_prompts)
                if not missing:
                    # Compliant — reset cooldown so re-violations re-fire.
                    if t.track_id in self._active_ppe:
                        self._active_ppe.pop(t.track_id, None)
                    continue
                last = self._active_ppe.get(t.track_id, -inf)
                if (timestamp - last) > self.cool_down_s:
                    events.append(
                        HazardEvent(
                            timestamp=timestamp,
                            track_id=t.track_id,
                            type="ppe_missing",
                            severity="warn",
                            details={"missing": missing},
                        )
                    )
                self._active_ppe[t.track_id] = timestamp

        self._all_events.extend(events)
        return events

    # ---------------------------------------------------------- ppe helpers
    @staticmethod
    def _crop(frame_rgb: np.ndarray, bbox: np.ndarray) -> np.ndarray | None:
        h, w = frame_rgb.shape[:2]
        x1, y1, x2, y2 = (int(v) for v in bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame_rgb[y1:y2, x1:x2]

    @staticmethod
    def _missing_ppe(dets, required: list[str]) -> list[str]:
        """Given a Detections object + required prompts, return missing items."""
        if dets is None or not hasattr(dets, "labels") or len(dets.labels) == 0:
            return list(required)
        seen: set[str] = set()
        for lab_idx in dets.labels:
            if 0 <= int(lab_idx) < len(dets.class_names):
                seen.add(dets.class_names[int(lab_idx)].lower())
        return [p for p in required if not any(p.lower() in s or s in p.lower() for s in seen)]

    # --------------------------------------------------------------- output
    def all_events(self) -> list[HazardEvent]:
        return list(self._all_events)

    def summary(self) -> dict[str, Any]:
        from collections import Counter

        ctype = Counter(e.type for e in self._all_events)
        cseverity = Counter(e.severity for e in self._all_events)
        per_track = Counter(e.track_id for e in self._all_events)
        return {
            "n_events": len(self._all_events),
            "by_type": dict(ctype),
            "by_severity": dict(cseverity),
            "by_track": dict(per_track),
        }


def draw_hazards(
    frame_bgr: np.ndarray,
    events: list[HazardEvent],
    tracks: list[PoseTrack],
) -> np.ndarray:
    """Render hazard banners on a BGR frame in place.

    Tracks listed in any event get a thick red/orange border + an
    ``!`` badge above the bbox. A bottom-of-frame caption summarizes the
    events fired this frame.
    """
    import cv2

    if not events:
        return frame_bgr

    flagged: dict[int, tuple[str, str]] = {}
    for e in events:
        flagged[e.track_id] = (e.type, e.severity)

    by_id = {t.track_id: t for t in tracks}
    for tid, (etype, sev) in flagged.items():
        t = by_id.get(tid)
        if t is None:
            continue
        color = (0, 0, 220) if sev == "danger" else (0, 140, 255)
        x1, y1, x2, y2 = (int(v) for v in t.bbox)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 4)
        text = f"! {etype}"
        cv2.rectangle(frame_bgr, (x1, max(y1 - 28, 0)), (x1 + 220, y1), color, -1)
        cv2.putText(
            frame_bgr, text, (x1 + 6, max(y1 - 8, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )
    return frame_bgr
