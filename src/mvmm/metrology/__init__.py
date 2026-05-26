"""Dimensional metrology — measure real-world distances/sizes from images.

Stack:
    calibration     — camera intrinsics + hand-eye + scale calibration
    segmentation    — SAM2-style prompted segmentation (or classical)
    depth           — Depth Anything v2 / stereo for absolute scale
    measure         — pixel→mm conversion, length/area/diameter primitives

Typical pipeline for automotive metrology:
    1. Acquire calibrated mono/stereo image.
    2. SAM2 mask the part (point or box prompt).
    3. Depth Anything monocular OR stereo block-matching → depth.
    4. Back-project mask boundary into 3D; fit primitives (line/circle/plane).
    5. Report measurements with uncertainty (calibration σ + depth σ).
"""

from __future__ import annotations

from mvmm.metrology.calibration import CameraIntrinsics, pixel_to_mm_scale
from mvmm.metrology.measure import circle_fit, dimension_from_mask, line_fit

__all__ = [
    "CameraIntrinsics",
    "circle_fit",
    "dimension_from_mask",
    "line_fit",
    "pixel_to_mm_scale",
]
