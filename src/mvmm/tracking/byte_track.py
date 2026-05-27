"""ByteTrack online tracker.

Reference:
    Zhang et al. "ByteTrack: Multi-Object Tracking by Associating Every
    Detection Box." ECCV 2022. https://arxiv.org/abs/2110.06864

Why ByteTrack matters for CCTV:
    * Online (per-frame, no future info)
    * Handles low-confidence boxes via second-stage association — recovers
      occluded objects that other trackers drop
    * No appearance model required — pure motion / IoU, so it survives
      ID-switching from re-identification model errors

We delegate the Kalman + Hungarian implementation to the ``supervision``
library (Roboflow), which carries a well-maintained ByteTrack port.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mvmm.tracking.detectors import Detections


@dataclass
class Track:
    """Single tracked object at one frame."""

    track_id: int
    bbox: np.ndarray  # (4,) xyxy
    score: float
    class_id: int
    class_name: str


class ByteTrackTracker:
    """Online ByteTrack wrapper.

    Args:
        frame_rate:          source FPS — controls Kalman noise.
        track_activation_threshold: minimum score for a new track.
        lost_track_buffer:   how many frames to keep a track alive after loss.
        minimum_matching_threshold: IoU threshold for first-stage association.
    """

    def __init__(
        self,
        frame_rate: int = 30,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
    ):
        try:
            import supervision as sv  # type: ignore
        except ImportError as e:
            raise ImportError("supervision is required for ByteTrack — `pip install supervision`") from e
        self._sv = sv
        self.tracker = sv.ByteTrack(
            frame_rate=frame_rate,
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
        )

    def update(self, detections: Detections) -> list[Track]:
        """Step the tracker by one frame."""
        if len(detections.boxes) == 0:
            sv_det = self._sv.Detections.empty()
        else:
            sv_det = self._sv.Detections(
                xyxy=detections.boxes,
                confidence=detections.scores,
                class_id=detections.labels,
            )
        tracked = self.tracker.update_with_detections(sv_det)
        out: list[Track] = []
        if len(tracked) == 0:
            return out
        for i in range(len(tracked)):
            tid = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else -1
            cls = int(tracked.class_id[i]) if tracked.class_id is not None else -1
            name = detections.class_names[cls] if 0 <= cls < len(detections.class_names) else "?"
            out.append(
                Track(
                    track_id=tid,
                    bbox=np.asarray(tracked.xyxy[i], dtype=np.float32),
                    score=float(tracked.confidence[i]) if tracked.confidence is not None else 0.0,
                    class_id=cls,
                    class_name=name,
                )
            )
        return out

    def reset(self) -> None:
        self.tracker.reset()
