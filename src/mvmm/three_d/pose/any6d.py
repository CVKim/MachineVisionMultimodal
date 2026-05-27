"""Any6D — model-free 6D pose estimation.

Reference:
    Liu et al. "Any6D: Model-free 6D Pose Estimation of Novel Objects."
    CVPR 2025. https://github.com/taeyeopl/Any6D

What makes it interesting for CCTV-based 3D:
    Unlike FoundationPose (which needs a CAD mesh or reference images),
    Any6D estimates 6D pose from a *single query image* — feed it a
    reference frame of the object and a new scene frame, and it produces
    the pose. Perfect for forensic / surveillance analysis where the CAD
    model is unknown.

Like FoundationPose, the upstream code has heavy CUDA / nvdiffrast deps.
This module wraps the integration so that downstream code stays uniform
even before the install is complete.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Any6DEstimate:
    transform: np.ndarray  # 4x4, units = mm in camera frame
    confidence: float
    method: str = "any6d"


class Any6DEstimator:
    """Wrapper around the Any6D codebase.

    Set ``MVMM_ANY6D_PATH`` to a cloned `taeyeopl/Any6D` repo root so the
    estimator can import the package at first use.
    """

    def __init__(self, intrinsics_3x3: np.ndarray, device: str = "cuda"):
        self.intrinsics = np.asarray(intrinsics_3x3, dtype=np.float64)
        self.device = device
        self._root = os.environ.get("MVMM_ANY6D_PATH")
        self._impl = None
        if self._root and Path(self._root).exists():
            try:
                self._impl = self._build()
            except Exception:
                self._impl = None

    def _build(self):
        import sys

        sys.path.insert(0, str(self._root))
        # The Any6D repo exposes a top-level Any6D class — names may shift
        # between commits, so we import lazily.
        from any6d import Any6D  # type: ignore

        return Any6D(K=self.intrinsics, device=self.device)

    def estimate(
        self,
        reference_rgb: np.ndarray,
        reference_depth: np.ndarray,
        query_rgb: np.ndarray,
        query_depth: np.ndarray,
    ) -> Any6DEstimate:
        if self._impl is None:
            raise NotImplementedError(
                "Any6D is not installed. Set MVMM_ANY6D_PATH to a clone of "
                "https://github.com/taeyeopl/Any6D and follow its install steps."
            )
        T = self._impl.estimate(reference_rgb, reference_depth, query_rgb, query_depth)
        return Any6DEstimate(transform=np.asarray(T, dtype=np.float64), confidence=1.0)
