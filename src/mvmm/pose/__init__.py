"""6D pose estimation & bin picking.

Modules:
    icp_classical       — ICP / point-to-plane refinement (Open3D-backed)
    foundation_pose     — wrapper for NVIDIA FoundationPose (CVPR'24 highlight)
    grasp               — sampling-based grasp candidate generator
    pipeline            — RGB(D) → mask → pose → grasp pose end-to-end
"""

from __future__ import annotations

from mvmm.pose.icp_classical import icp_refine
from mvmm.pose.pipeline import PosePipeline

__all__ = ["PosePipeline", "icp_refine"]
