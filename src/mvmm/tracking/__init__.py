"""CCTV / video tracking stack.

A pluggable pipeline:

    frames ─▶ detector ─▶ tracker ─▶ (optional: ReID, SAM2 masks) ─▶ analytics

Submodules:
    detectors     — YOLOv8/v11, RT-DETR, GroundingDINO open-vocab
    byte_track    — ByteTrack online tracker (via `supervision`)
    bot_sort      — BoT-SORT with appearance / Kalman fusion
    sam2_video    — SAM2 video predictor (memory-aware mask tracking)
    reid          — OSNet / TransReID appearance embeddings
    analytics     — zone counting, dwell time, line crossing
    pipeline      — end-to-end detect→track→annotate→stats
"""

from __future__ import annotations

from mvmm.tracking.byte_track import ByteTrackTracker
from mvmm.tracking.detectors import build_detector
from mvmm.tracking.pipeline import TrackingPipeline, TrackingResult

__all__ = ["ByteTrackTracker", "TrackingPipeline", "TrackingResult", "build_detector"]
