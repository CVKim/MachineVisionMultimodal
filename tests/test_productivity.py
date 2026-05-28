"""Productivity / bucketed time accounting tests."""

from __future__ import annotations

import numpy as np

from mvmm.tracking.pose_activity import ActivityClassifier, PoseTrack


def _make_track(tid: int, kps: np.ndarray, bbox: np.ndarray) -> PoseTrack:
    return PoseTrack(
        track_id=tid,
        bbox=bbox,
        keypoints=kps.astype(np.float32),
        kp_conf=np.full(17, 0.9, dtype=np.float32),
        score=0.9,
    )


def _standing_kps(cx: float = 100.0, cy: float = 200.0, scale: float = 100.0) -> np.ndarray:
    kps = np.zeros((17, 2), dtype=np.float32)
    # Full skeleton — see tests/test_pose_activity.py for the canonical version.
    kps[0] = [cx, cy - 0.45 * scale]
    kps[1] = [cx - 0.03 * scale, cy - 0.47 * scale]
    kps[2] = [cx + 0.03 * scale, cy - 0.47 * scale]
    kps[3] = [cx - 0.07 * scale, cy - 0.45 * scale]
    kps[4] = [cx + 0.07 * scale, cy - 0.45 * scale]
    kps[5] = [cx - 0.18 * scale, cy - 0.30 * scale]
    kps[6] = [cx + 0.18 * scale, cy - 0.30 * scale]
    kps[7] = [cx - 0.22 * scale, cy - 0.10 * scale]
    kps[8] = [cx + 0.22 * scale, cy - 0.10 * scale]
    kps[9] = [cx - 0.22 * scale, cy + 0.05 * scale]
    kps[10] = [cx + 0.22 * scale, cy + 0.05 * scale]
    kps[11] = [cx - 0.12 * scale, cy + 0.10 * scale]
    kps[12] = [cx + 0.12 * scale, cy + 0.10 * scale]
    kps[13] = [cx - 0.12 * scale, cy + 0.30 * scale]
    kps[14] = [cx + 0.12 * scale, cy + 0.30 * scale]
    kps[15] = [cx - 0.12 * scale, cy + 0.45 * scale]
    kps[16] = [cx + 0.12 * scale, cy + 0.45 * scale]
    return kps


def _bbox(kps: np.ndarray) -> np.ndarray:
    pad = 8.0
    return np.array(
        [kps[:, 0].min() - pad, kps[:, 1].min() - pad, kps[:, 0].max() + pad, kps[:, 1].max() + pad],
        dtype=np.float32,
    )


def test_buckets_default_to_one_bucket_when_max_t_small():
    clf = ActivityClassifier(fps=10.0)
    kps = _standing_kps()
    bbox = _bbox(kps)
    for i in range(5):
        clf.update([_make_track(1, kps, bbox)], timestamp=i * 0.1)
    buckets = clf.time_in_state_by_bucket(bucket_seconds=10.0)
    assert 1 in buckets and len(buckets[1]) == 1
    # The total over the single bucket should match the unbucketed total.
    total_unbucketed = sum(clf.time_in_state()[1].values())
    total_bucket = sum(v for k, v in buckets[1][0].items() if k not in {"bucket_start_s", "bucket_end_s"})
    assert abs(total_unbucketed - total_bucket) < 1e-4


def test_buckets_split_intervals_across_boundaries():
    clf = ActivityClassifier(fps=10.0)
    kps = _standing_kps()
    bbox = _bbox(kps)
    # Two state-log entries spanning a 5s+ gap.
    clf.update([_make_track(1, kps, bbox)], timestamp=0.0)
    clf.update([_make_track(1, kps, bbox)], timestamp=4.0)
    clf.update([_make_track(1, kps, bbox)], timestamp=12.0)  # 8-second interval
    buckets = clf.time_in_state_by_bucket(bucket_seconds=5.0)
    # Should have at least 3 buckets (0-5, 5-10, 10-15).
    assert len(buckets[1]) >= 3
    # Total time across buckets matches sum of inter-frame deltas (4 + 8 = 12s).
    total = sum(v for b in buckets[1] for k, v in b.items() if k not in {"bucket_start_s", "bucket_end_s"})
    assert abs(total - 12.0) < 1e-3


def test_productivity_per_track_returns_fraction_in_unit_interval():
    clf = ActivityClassifier(fps=10.0)
    kps = _standing_kps()
    bbox = _bbox(kps)
    for i in range(20):
        clf.update([_make_track(1, kps, bbox)], timestamp=i * 0.1)
    prod = clf.productivity_per_track()
    assert 1 in prod
    assert 0.0 <= prod[1] <= 1.0
