"""End-to-end bin-picking pipeline (skeleton).

Steps (per frame):
    RGB + Depth → segment (SAM2/classical) → 6D pose (FoundationPose/ICP)
                                          → top-K grasp candidates → pick

This file glues the three subsystems together. The actual choice of
backends is configurable so you can run a CPU-only sanity test (classical
seg + ICP + antipodal grasps) or a fully-loaded GPU pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class BinPickFrame:
    rgb: np.ndarray  # HxWx3 uint8
    depth_mm: np.ndarray  # HxW float
    intrinsics_3x3: np.ndarray  # 3x3


@dataclass
class BinPickResult:
    mask: np.ndarray
    pose: np.ndarray | None
    grasps: list[Any]


class PosePipeline:
    """Pluggable pose + grasp pipeline.

    Args:
        segmenter:       callable(rgb, points=..., box=...) -> mask
        pose_estimator:  callable(rgb, depth, mask) -> 4x4 transform or None
        grasp_sampler:   callable(points, normals) -> list[GraspCandidate]
    """

    def __init__(self, segmenter: Any, pose_estimator: Any | None, grasp_sampler: Any):
        self.segmenter = segmenter
        self.pose_estimator = pose_estimator
        self.grasp_sampler = grasp_sampler

    def __call__(
        self,
        frame: BinPickFrame,
        seed_point: tuple[int, int] | None = None,
    ) -> BinPickResult:
        prompt_points = None
        if seed_point is not None:
            prompt_points = [(seed_point[0], seed_point[1], 1)]
        mask = self.segmenter(frame.rgb, points=prompt_points)

        pose_T = None
        if self.pose_estimator is not None:
            try:
                pose_T = self.pose_estimator.estimate(frame.rgb, frame.depth_mm, mask).transform
            except NotImplementedError:
                pose_T = None

        # Build a quick point cloud from masked depth for grasp sampling.
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return BinPickResult(mask=mask, pose=pose_T, grasps=[])

        z = frame.depth_mm[ys, xs]
        fx, fy = frame.intrinsics_3x3[0, 0], frame.intrinsics_3x3[1, 1]
        cx, cy = frame.intrinsics_3x3[0, 2], frame.intrinsics_3x3[1, 2]
        X = (xs - cx) * z / fx
        Y = (ys - cy) * z / fy
        points = np.stack([X, Y, z], axis=-1)

        # Cheap surface normals via local PCA — for a baseline, use z-up assumption.
        normals = np.tile(np.array([0, 0, -1.0]), (points.shape[0], 1))
        grasps = self.grasp_sampler(points, normals)
        return BinPickResult(mask=mask, pose=pose_T, grasps=grasps)
