"""ActivityClassifier rule tests — no pose model needed."""

from __future__ import annotations

import numpy as np

from mvmm.tracking.pose_activity import (
    KP_LELBOW,
    KP_LHIP,
    KP_LKNEE,
    KP_LSHOULDER,
    KP_LWRIST,
    KP_RELBOW,
    KP_RHIP,
    KP_RKNEE,
    KP_RSHOULDER,
    KP_RWRIST,
    ActivityClassifier,
    PoseTrack,
)


def _make_track(track_id: int, kps: np.ndarray, bbox: np.ndarray) -> PoseTrack:
    conf = np.full(17, 0.9, dtype=np.float32)
    return PoseTrack(track_id=track_id, bbox=bbox, keypoints=kps.astype(np.float32), kp_conf=conf, score=0.9)


def _standing_kps(cx: float = 100.0, cy: float = 200.0, scale: float = 100.0) -> np.ndarray:
    """A roughly upright COCO 17-keypoint skeleton centered at (cx, cy).

    All 17 keypoints are set so a bbox built from kps min/max actually
    reflects body extent (otherwise unset ``(0, 0)`` entries inflate the
    bbox height and skew motion-ratio thresholds — that exact bug bit a
    previous version of this test).
    """
    kps = np.zeros((17, 2), dtype=np.float32)
    kps[0] = [cx, cy - 0.45 * scale]  # nose
    kps[1] = [cx - 0.03 * scale, cy - 0.47 * scale]
    kps[2] = [cx + 0.03 * scale, cy - 0.47 * scale]
    kps[3] = [cx - 0.07 * scale, cy - 0.45 * scale]
    kps[4] = [cx + 0.07 * scale, cy - 0.45 * scale]
    kps[KP_LSHOULDER] = [cx - 0.18 * scale, cy - 0.30 * scale]
    kps[KP_RSHOULDER] = [cx + 0.18 * scale, cy - 0.30 * scale]
    kps[KP_LELBOW] = [cx - 0.22 * scale, cy - 0.10 * scale]
    kps[KP_RELBOW] = [cx + 0.22 * scale, cy - 0.10 * scale]
    kps[KP_LWRIST] = [cx - 0.22 * scale, cy + 0.05 * scale]
    kps[KP_RWRIST] = [cx + 0.22 * scale, cy + 0.05 * scale]
    kps[KP_LHIP] = [cx - 0.12 * scale, cy + 0.10 * scale]
    kps[KP_RHIP] = [cx + 0.12 * scale, cy + 0.10 * scale]
    kps[KP_LKNEE] = [cx - 0.12 * scale, cy + 0.30 * scale]
    kps[KP_RKNEE] = [cx + 0.12 * scale, cy + 0.30 * scale]
    kps[15] = [cx - 0.12 * scale, cy + 0.45 * scale]
    kps[16] = [cx + 0.12 * scale, cy + 0.45 * scale]
    return kps


def _bbox_for(kps: np.ndarray) -> np.ndarray:
    pad = 8.0
    return np.array(
        [kps[:, 0].min() - pad, kps[:, 1].min() - pad, kps[:, 0].max() + pad, kps[:, 1].max() + pad],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
def test_classifier_too_short_history_is_unknown():
    clf = ActivityClassifier(fps=10.0)
    kps = _standing_kps()
    bbox = _bbox_for(kps)
    clf.update([_make_track(1, kps, bbox)], timestamp=0.0)
    clf.update([_make_track(1, kps, bbox)], timestamp=0.1)
    assert clf.histories[1].state_log[-1][1] == "unknown"


def test_classifier_idle_when_keypoints_static():
    clf = ActivityClassifier(fps=10.0, window_frames=10, smooth_frames=4)
    kps = _standing_kps()
    bbox = _bbox_for(kps)
    for i in range(10):
        clf.update([_make_track(1, kps, bbox)], timestamp=i / 10.0)
    states = [s for _, s in clf.histories[1].state_log[-4:]]
    assert "idle" in states


def test_classifier_walking_when_hip_translates():
    clf = ActivityClassifier(fps=10.0, window_frames=10, smooth_frames=4)
    for i in range(10):
        # Translate the whole skeleton horizontally by 12 px/frame so the
        # window-wide hip displacement clearly clears the 0.35 * bbox_h threshold.
        kps = _standing_kps(cx=100.0 + 12.0 * i)
        bbox = _bbox_for(kps)
        clf.update([_make_track(1, kps, bbox)], timestamp=i / 10.0)
    states = [s for _, s in clf.histories[1].state_log[-4:]]
    assert "walking" in states


def test_classifier_working_when_only_wrists_move():
    clf = ActivityClassifier(fps=10.0, window_frames=10, smooth_frames=4)
    base = _standing_kps()
    bbox = _bbox_for(base)
    for i in range(10):
        kps = base.copy()
        # Horizontal wrist oscillation — keeps wrists *above* the hip
        # (so the "lifting" branch doesn't fire) while creating large
        # per-frame deltas that clear the working threshold.
        kps[KP_LWRIST, 0] += 12.0 if i % 2 == 0 else -12.0
        kps[KP_RWRIST, 0] += -12.0 if i % 2 == 0 else 12.0
        clf.update([_make_track(1, kps, bbox)], timestamp=i / 10.0)
    states = [s for _, s in clf.histories[1].state_log[-4:]]
    assert "working" in states


def test_time_in_state_accumulates_correctly():
    clf = ActivityClassifier(fps=10.0, window_frames=4, smooth_frames=2)
    kps = _standing_kps()
    bbox = _bbox_for(kps)
    for i in range(6):
        clf.update([_make_track(1, kps, bbox)], timestamp=i / 10.0)
    per_track = clf.time_in_state()
    assert 1 in per_track
    total = sum(per_track[1].values())
    # Five 0.1-s gaps -> ~0.5s total accounted for.
    assert 0.4 < total < 0.6


def test_total_time_aggregates_across_tracks():
    clf = ActivityClassifier(fps=10.0, window_frames=4, smooth_frames=2)
    kps_a = _standing_kps(cx=100.0)
    kps_b = _standing_kps(cx=300.0)
    bbox_a, bbox_b = _bbox_for(kps_a), _bbox_for(kps_b)
    for i in range(6):
        clf.update(
            [_make_track(1, kps_a, bbox_a), _make_track(2, kps_b, bbox_b)],
            timestamp=i / 10.0,
        )
    totals = clf.total_time_in_state()
    assert sum(totals.values()) > 0.9  # ~0.5 + 0.5 = 1.0s combined
