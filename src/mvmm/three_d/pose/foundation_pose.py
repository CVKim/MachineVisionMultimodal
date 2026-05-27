"""FoundationPose wrapper — model-agnostic 6D pose & tracking.

Reference:
    Wen et al. "FoundationPose: Unified 6D Pose Estimation and Tracking
    of Novel Objects." CVPR 2024 (highlight).
    https://github.com/NVlabs/FoundationPose

This wrapper provides a thin API that mirrors our PosePipeline interface.
The actual FoundationPose codebase has a heavy native build (PyTorch3D,
custom CUDA ops); installing it is out-of-scope for this repo's CI but
the wrapper makes integration straightforward when you set it up locally.

Recommended integration (see docs/ROADMAP.md):
    1. Clone NVlabs/FoundationPose under third_party/
    2. Follow its install instructions inside the container
    3. Set MVMM_FOUNDATIONPOSE_PATH env var to its repo root
    4. Use ``FoundationPoseEstimator(cad_mesh_path=..., intrinsics=...)``
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class PoseEstimate:
    """6D pose result."""

    transform: np.ndarray  # 4x4 in camera frame, units = mm
    confidence: float
    object_id: str | None = None


class FoundationPoseEstimator:
    """Wrapper class. Falls back to NotImplementedError if FoundationPose
    is not installed locally. See ``docs/ROADMAP.md`` for setup steps.
    """

    def __init__(
        self,
        cad_mesh_path: str | Path,
        intrinsics_3x3: np.ndarray,
        device: str = "cuda",
        checkpoint_dir: str | Path | None = None,
    ):
        self.cad_mesh_path = Path(cad_mesh_path)
        self.intrinsics = np.asarray(intrinsics_3x3, dtype=np.float64)
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self._fp_root = os.environ.get("MVMM_FOUNDATIONPOSE_PATH")
        self._impl = None
        if self._fp_root and Path(self._fp_root).exists():
            try:
                self._impl = self._build_impl()
            except Exception:
                self._impl = None

    def _build_impl(self):
        """Construct the FoundationPose model when its source tree is available.

        We *do not* import at module load — the FoundationPose package can
        be heavy to import (compiles CUDA on first use).
        """
        import sys

        sys.path.insert(0, str(self._fp_root))  # type: ignore[arg-type]
        from estimater import FoundationPose  # type: ignore

        return FoundationPose(
            mesh_file=str(self.cad_mesh_path),
            scorer=None,
            refiner=None,
            debug=0,
            glctx=None,
        )

    def estimate(
        self,
        rgb: np.ndarray,
        depth_mm: np.ndarray,
        mask: np.ndarray,
    ) -> PoseEstimate:
        if self._impl is None:
            raise NotImplementedError(
                "FoundationPose not available. Set MVMM_FOUNDATIONPOSE_PATH to the "
                "cloned NVlabs/FoundationPose repo and install its requirements. "
                "Alternative: use mvmm.pose.icp_classical for refinement only."
            )
        pose = self._impl.register(K=self.intrinsics, rgb=rgb, depth=depth_mm, ob_mask=mask)
        return PoseEstimate(transform=np.asarray(pose, dtype=np.float64), confidence=1.0)

    def track(self, rgb: np.ndarray, depth_mm: np.ndarray) -> PoseEstimate:
        if self._impl is None:
            raise NotImplementedError("FoundationPose not available — see estimate()")
        pose = self._impl.track_one(rgb=rgb, depth=depth_mm, K=self.intrinsics)
        return PoseEstimate(transform=np.asarray(pose, dtype=np.float64), confidence=1.0)
