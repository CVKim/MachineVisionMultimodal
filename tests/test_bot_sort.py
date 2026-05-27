"""BoT-SORT composition test — no real frames, just a fake ReID + ByteTrack stub."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from mvmm.tracking.bot_sort import BoTSORTTracker
from mvmm.tracking.byte_track import Track
from mvmm.tracking.detectors import Detections


class _StubReID:
    """Returns deterministic embeddings from BGR mean for unit-testing."""

    def embed(self, crops):
        out = []
        for c in crops:
            v = np.array([c.mean(axis=(0, 1))[0] / 255.0, c.std() / 255.0])
            out.append(v / (np.linalg.norm(v) + 1e-9))
        return np.stack(out).astype(np.float32) if out else np.zeros((0, 2), dtype=np.float32)


def test_botsort_initializes_without_supervision():
    """Just verify the class can be constructed without supervision via mock."""
    tracker = BoTSORTTracker.__new__(BoTSORTTracker)
    tracker.reid = _StubReID()
    tracker.appearance_dist_threshold = 0.35
    tracker.appearance_alpha = 0.9
    tracker._byte = MagicMock()
    tracker._byte.update = MagicMock(return_value=[])
    tracker._states = {}
    tracker._frame_idx = 0

    # No detections -> no tracks; method just GCs and returns empty.
    out = tracker.update(
        Detections(
            boxes=np.zeros((0, 4), dtype=np.float32),
            scores=np.zeros((0,), dtype=np.float32),
            labels=np.zeros((0,), dtype=np.int64),
            class_names=[],
        ),
        frame_rgb=np.zeros((50, 50, 3), dtype=np.uint8),
    )
    assert out == []


def test_botsort_remaps_to_lost_track_by_appearance():
    """A new ByteTrack id with matching appearance to a previously seen id
    should be re-mapped to the original id."""
    tracker = BoTSORTTracker.__new__(BoTSORTTracker)
    tracker.reid = _StubReID()
    tracker.appearance_dist_threshold = 0.99  # very permissive
    tracker.appearance_alpha = 0.9
    tracker._byte = MagicMock()
    tracker._states = {}
    tracker._frame_idx = 0

    frame = np.full((40, 80, 3), 200, dtype=np.uint8)
    dets = Detections(
        boxes=np.array([[10, 10, 30, 30]], dtype=np.float32),
        scores=np.array([0.9], dtype=np.float32),
        labels=np.array([0]),
        class_names=["person"],
    )

    # Frame 1: ByteTrack says track_id=1
    tracker._byte.update = MagicMock(
        return_value=[
            Track(track_id=1, bbox=np.array([10, 10, 30, 30]), score=0.9, class_id=0, class_name="person")
        ]
    )
    out1 = tracker.update(dets, frame_rgb=frame)
    assert out1[0].track_id == 1

    # Frame 2: ByteTrack lost it and now says track_id=2 for the *same* appearance
    tracker._byte.update = MagicMock(
        return_value=[
            Track(track_id=2, bbox=np.array([10, 10, 30, 30]), score=0.9, class_id=0, class_name="person")
        ]
    )
    out2 = tracker.update(dets, frame_rgb=frame)
    # With a permissive appearance threshold, BoT-SORT should re-map 2 -> 1.
    assert out2[0].track_id == 1
