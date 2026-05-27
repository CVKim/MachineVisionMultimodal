"""Classical ICP refinement using Open3D.

For full deep-learning pipelines see ``foundation_pose.py``; ICP is
still indispensable for fine alignment after a coarse DL pose.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ICPResult:
    transform: np.ndarray  # 4x4
    fitness: float
    inlier_rmse: float


def icp_refine(
    source_points: np.ndarray,
    target_points: np.ndarray,
    init_transform: np.ndarray | None = None,
    max_corr_dist: float = 5.0,  # mm
    point_to_plane: bool = True,
) -> ICPResult:
    """Refine an initial guess with ICP.

    Args:
        source_points: (N, 3) — model points.
        target_points: (M, 3) — observed scene points (e.g. from depth).
        init_transform: 4x4 initial pose; identity if None.
        max_corr_dist: nearest-neighbor cutoff (units of the point cloud).
        point_to_plane: use point-to-plane (requires normals on target).
    """
    try:
        import open3d as o3d  # type: ignore
    except ImportError as e:
        raise ImportError("open3d is required for icp_refine — `pip install open3d`") from e

    src = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(np.asarray(source_points, dtype=np.float64))
    tgt = o3d.geometry.PointCloud()
    tgt.points = o3d.utility.Vector3dVector(np.asarray(target_points, dtype=np.float64))

    if point_to_plane:
        tgt.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=max_corr_dist * 2, max_nn=30)
        )
        method: Any = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        method = o3d.pipelines.registration.TransformationEstimationPointToPoint()

    init = np.eye(4) if init_transform is None else np.asarray(init_transform, dtype=np.float64)
    reg = o3d.pipelines.registration.registration_icp(src, tgt, max_corr_dist, init, method)
    return ICPResult(
        transform=np.asarray(reg.transformation),
        fitness=float(reg.fitness),
        inlier_rmse=float(reg.inlier_rmse),
    )
