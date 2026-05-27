"""3D perception: depth, pose, metrology, point cloud, scene reconstruction.

Submodules:
    depth/                — Depth Anything v2, MoGe, Marigold, stereo SGBM
    pose/                 — FoundationPose, Any6D, classical ICP, grasping
    metrology/            — calibration, prompted segmentation, primitive measurements
    reconstruction.py     — depth → point cloud (PLY export)
    gaussian_splatting.py — 3DGS wrapper (gsplat / Inria backends)
"""

from __future__ import annotations

from mvmm.three_d.reconstruction import back_project, save_ply

__all__ = ["back_project", "save_ply"]
