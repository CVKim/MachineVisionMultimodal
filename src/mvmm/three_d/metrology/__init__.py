"""Dimensional metrology — measure real-world distances/sizes from images."""

from __future__ import annotations

from mvmm.three_d.metrology.calibration import CameraIntrinsics, pixel_to_mm_scale
from mvmm.three_d.metrology.measure import (
    CircleFit,
    DimensionResult,
    LineFit,
    circle_fit,
    dimension_from_mask,
    line_fit,
)
from mvmm.three_d.metrology.segmentation import (
    ClassicalSegmenter,
    SAM2Segmenter,
    build_segmenter,
)

__all__ = [
    "CameraIntrinsics",
    "CircleFit",
    "ClassicalSegmenter",
    "DimensionResult",
    "LineFit",
    "SAM2Segmenter",
    "build_segmenter",
    "circle_fit",
    "dimension_from_mask",
    "line_fit",
    "pixel_to_mm_scale",
]
