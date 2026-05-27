"""Depth estimation backends — monocular foundation models + classical stereo."""

from __future__ import annotations

from mvmm.three_d.depth.depth import DepthAnythingV2, StereoSGBM, build_depth_estimator
from mvmm.three_d.depth.stereo_synth import synthesize_right_view

__all__ = ["DepthAnythingV2", "StereoSGBM", "build_depth_estimator", "synthesize_right_view"]
