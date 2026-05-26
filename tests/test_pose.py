"""Pose / grasp module tests — no GPU."""

from __future__ import annotations

import numpy as np

from mvmm.pose.grasp import antipodal_grasps


def test_antipodal_grasps_on_synthetic_pair():
    # Two opposing points on a 50mm wide bar.
    points = np.array(
        [
            [0.0, 0.0, 100.0],
            [50.0, 0.0, 100.0],
        ]
    )
    normals = np.array(
        [
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    grasps = antipodal_grasps(points, normals, gripper_width_mm=80.0, min_width_mm=5.0, max_candidates=4)
    assert len(grasps) >= 1
    g = grasps[0]
    assert 40 < g.width < 60
    assert g.score > 0
