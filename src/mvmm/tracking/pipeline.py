"""End-to-end CCTV tracking pipeline.

    video.mp4 ─▶ frame loop ─▶ detector ─▶ tracker ─▶ analytics ─▶ annotated video

Designed to be **streaming-friendly** (process frame-by-frame without
loading the whole video) and to write incremental JSON + MP4 outputs so
you can inspect results mid-run.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from mvmm.tracking.analytics import DwellTimer, LineCounter, PolygonZone
from mvmm.tracking.byte_track import ByteTrackTracker, Track
from mvmm.tracking.detectors import Detections


@dataclass
class TrackingResult:
    frame_idx: int
    tracks: list[Track]


@dataclass
class TrackingStats:
    n_frames: int = 0
    n_detections_total: int = 0
    track_ids_seen: set[int] = field(default_factory=set)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_frames": self.n_frames,
            "n_detections_total": self.n_detections_total,
            "n_unique_track_ids": len(self.track_ids_seen),
        }


def _iter_video_frames(path: Path) -> Iterator[tuple[int, np.ndarray, float]]:
    """Yield (frame_idx, frame_bgr, fps) until end of stream."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield idx, frame, fps
            idx += 1
    finally:
        cap.release()


def _draw_tracks(frame_bgr: np.ndarray, tracks: list[Track]) -> np.ndarray:
    """Render tracks onto BGR frame with deterministic per-ID colors."""
    out = frame_bgr.copy()
    for t in tracks:
        rng = np.random.default_rng(t.track_id * 9301 + 49297)
        color = tuple(int(c) for c in rng.integers(64, 255, size=3))
        x1, y1, x2, y2 = (int(v) for v in t.bbox)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"#{t.track_id} {t.class_name} {t.score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def _draw_zone(frame_bgr: np.ndarray, zone: PolygonZone, count: int) -> np.ndarray:
    overlay = frame_bgr.copy()
    pts = zone.polygon.astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(overlay, [pts], (0, 200, 0))
    out = cv2.addWeighted(overlay, 0.18, frame_bgr, 0.82, 0)
    cv2.polylines(out, [pts], isClosed=True, color=(0, 200, 0), thickness=2)
    cx, cy = pts.mean(axis=0).ravel().astype(int)
    cv2.putText(out, f"{zone.name}: {count}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
    return out


class TrackingPipeline:
    """Compose a detector + tracker + optional analytics.

    Args:
        detector:        callable(image_rgb, classes) -> Detections
        tracker:         ByteTrackTracker (or compatible)
        classes:         list of class names to keep (None = all)
        score_threshold: min score after detection
        zones:           list of PolygonZone for counting
        line_counters:   list of LineCounter for entry/exit
        dwell_timers:    list of DwellTimer
    """

    def __init__(
        self,
        detector: Callable,
        tracker: ByteTrackTracker,
        classes: list[str] | None = None,
        score_threshold: float = 0.25,
        zones: list[PolygonZone] | None = None,
        line_counters: list[LineCounter] | None = None,
        dwell_timers: list[DwellTimer] | None = None,
    ):
        self.detector = detector
        self.tracker = tracker
        self.classes = classes
        self.score_threshold = score_threshold
        self.zones = zones or []
        self.line_counters = line_counters or []
        self.dwell_timers = dwell_timers or []

    # ----------------------------- single-frame ----------------------------
    def process_frame(self, frame_rgb: np.ndarray) -> list[Track]:
        dets: Detections = self.detector(frame_rgb, classes=self.classes)
        dets = dets.filter_by_score(self.score_threshold)
        tracks = self.tracker.update(dets)
        for line in self.line_counters:
            line.update(tracks)
        for dwell in self.dwell_timers:
            dwell.update(tracks)
        return tracks

    # ------------------------------- video ---------------------------------
    def process_video(
        self,
        input_path: str | Path,
        output_video: str | Path | None = None,
        output_json: str | Path | None = None,
        show_progress: bool = True,
    ) -> TrackingStats:
        input_path = Path(input_path)
        writer: cv2.VideoWriter | None = None
        records: list[dict[str, Any]] = []
        stats = TrackingStats()

        try:
            from tqdm import tqdm  # type: ignore

            wrap = tqdm if show_progress else (lambda x, **k: x)
        except ImportError:
            wrap = lambda x, **k: x  # noqa: E731

        frames = list(_iter_video_frames(input_path))
        if not frames:
            raise RuntimeError(f"No frames decoded from {input_path}")
        _, first_frame, fps = frames[0]
        h, w = first_frame.shape[:2]

        if output_video is not None:
            output_video = Path(output_video)
            output_video.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_video), fourcc, fps, (w, h))

        for idx, frame_bgr, _fps in wrap(frames, desc="track"):
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            tracks = self.process_frame(frame_rgb)

            stats.n_frames += 1
            stats.n_detections_total += len(tracks)
            for t in tracks:
                stats.track_ids_seen.add(t.track_id)

            records.append(
                {
                    "frame": idx,
                    "tracks": [
                        {
                            "id": t.track_id,
                            "bbox": [float(v) for v in t.bbox.tolist()],
                            "class": t.class_name,
                            "score": t.score,
                        }
                        for t in tracks
                    ],
                    "zones": {z.name: z.count(tracks) for z in self.zones},
                }
            )

            if writer is not None:
                vis = _draw_tracks(frame_bgr, tracks)
                for z in self.zones:
                    vis = _draw_zone(vis, z, z.count(tracks))
                writer.write(vis)

        if writer is not None:
            writer.release()
        if output_json is not None:
            output_json = Path(output_json)
            output_json.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "stats": stats.as_dict(),
                "lines": [{"name": lc.name, "in": lc.in_count, "out": lc.out_count} for lc in self.line_counters],
                "dwell_seconds": [{"name": d.zone.name, "values": d.seconds(fps)} for d in self.dwell_timers],
                "per_frame": records,
            }
            with output_json.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        return stats
