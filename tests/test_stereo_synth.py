"""Stereo-from-monocular synthesis correctness checks."""

from __future__ import annotations

import numpy as np

from mvmm.three_d.depth.stereo_synth import synthesize_right_view


def test_synth_right_shapes_and_disparity():
    h, w = 32, 64
    rgb = (np.random.rand(h, w, 3) * 255).astype(np.uint8)
    depth = np.full((h, w), 2.0, dtype=np.float32)  # constant depth
    right, disp = synthesize_right_view(rgb, depth, focal_px=400.0, baseline_mm=60.0)
    assert right.shape == rgb.shape and right.dtype == np.uint8
    assert disp.shape == (h, w)
    # All pixels at the same depth → all disparities equal.
    assert np.allclose(disp, disp.mean(), atol=1e-4)


def test_synth_zero_depth_handled():
    h, w = 16, 16
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    depth = np.zeros((h, w), dtype=np.float32)  # degenerate
    right, disp = synthesize_right_view(rgb, depth, focal_px=400.0, baseline_mm=60.0)
    assert right.shape == rgb.shape
    assert np.isfinite(disp).all()
