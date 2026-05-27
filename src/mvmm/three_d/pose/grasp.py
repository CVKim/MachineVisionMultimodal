"""Antipodal grasp sampler — classical baseline for parallel-jaw grippers.

For learned grasping (GraspNet, AnyGrasp) plug into ``grasp_dl.py`` later.

Algorithm:
    1. Estimate normals on the segmented object surface.
    2. For each pair of antipodal candidate points (n_i · n_j ≈ -1),
       check if the gripper width fits and the line between them is
       free of clutter.
    3. Rank by force-closure proxy (alignment + width margin).

This is the kind of pipeline you usually start from in semi/auto bin picking
before training a deep grasp model on your own labels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GraspCandidate:
    p1: np.ndarray  # (3,)
    p2: np.ndarray  # (3,)
    axis: np.ndarray  # (3,) — unit vector along gripper closing direction
    width: float
    score: float


def antipodal_grasps(
    points: np.ndarray,
    normals: np.ndarray,
    gripper_width_mm: float = 80.0,
    min_width_mm: float = 5.0,
    antipodal_thresh: float = 0.85,
    max_candidates: int = 64,
) -> list[GraspCandidate]:
    """Find antipodal grasp pairs from a labeled point cloud with normals.

    Args:
        points:  (N, 3) point cloud (mm).
        normals: (N, 3) unit normals.
        gripper_width_mm: maximum opening of the gripper.
        min_width_mm: minimum object thickness to consider.
        antipodal_thresh: |n_i · -n_j| > threshold counts as antipodal.
        max_candidates: cap on returned candidates.
    """
    if len(points) < 2:
        return []
    pts = np.asarray(points, dtype=np.float64)
    nrm = np.asarray(normals, dtype=np.float64)
    nrm = nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9)

    # Random pair sampling — fast and good enough for a baseline.
    rng = np.random.default_rng(0)
    n = len(pts)
    candidates: list[GraspCandidate] = []
    tries = min(max_candidates * 20, n * n)
    i_arr = rng.integers(0, n, size=tries)
    j_arr = rng.integers(0, n, size=tries)
    for i, j in zip(i_arr, j_arr, strict=False):
        if i == j:
            continue
        v = pts[j] - pts[i]
        w = float(np.linalg.norm(v))
        if not (min_width_mm <= w <= gripper_width_mm):
            continue
        axis = v / (w + 1e-9)
        align = float(np.abs(nrm[i] @ -nrm[j]))
        if align < antipodal_thresh:
            continue
        # Score: antipodal alignment + how well normals oppose the closing axis.
        push_i = float(np.abs(nrm[i] @ axis))
        push_j = float(np.abs(nrm[j] @ -axis))
        score = align * 0.5 + (push_i + push_j) * 0.25
        candidates.append(GraspCandidate(p1=pts[i], p2=pts[j], axis=axis, width=w, score=score))
        if len(candidates) >= max_candidates:
            break

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:max_candidates]
