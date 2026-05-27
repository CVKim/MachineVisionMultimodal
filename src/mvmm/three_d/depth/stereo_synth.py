"""Synthesize a virtual right-eye view from a single (mono) image + depth.

Why this is useful:
    * Test stereo block-matching / disparity refinement code paths
      without a real stereo rig.
    * Generate quick training data for depth-supervised learners by
      pairing the original image with the synthesized stereo pair.
    * Validate calibration math (baseline, focal) — the disparity
      after synthesis should match the ground-truth depth.

Algorithm:
    For pixel (u, v) with depth Z, the corresponding pixel in the
    right view at baseline B and focal length f is at

        u_r = u - (f * B / Z)

    For each output column we read from the *closest* foreground source.
    Holes (occlusions) are filled by horizontal interpolation.
"""

from __future__ import annotations

import cv2
import numpy as np


def synthesize_right_view(
    image_rgb: np.ndarray,
    depth_m: np.ndarray,
    focal_px: float = 700.0,
    baseline_mm: float = 60.0,
    fill_holes: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the right-eye view.

    Args:
        image_rgb:   HxWx3 uint8 source image (the "left" view).
        depth_m:     HxW float, depth in meters (Depth Anything output works
                     fine if you treat its arbitrary units as metric).
        focal_px:    nominal focal length in pixels.
        baseline_mm: stereo baseline in millimeters.

    Returns:
        (right_view_rgb HxWx3 uint8, disparity_px HxW float)
    """
    h, w = depth_m.shape
    baseline_m = baseline_mm / 1000.0
    z = np.maximum(depth_m, 1e-6).astype(np.float32)
    disp = (focal_px * baseline_m) / z  # pixel disparity

    # Forward-warp the image columns by -disparity.
    right = np.zeros_like(image_rgb)
    occ_mask = np.ones((h, w), dtype=np.uint8) * 255  # 255 = hole

    # Sort by depth so far surfaces draw first; near surfaces overwrite.
    order = np.argsort(-z, axis=1)  # per-row, far -> near
    rows = np.arange(h)[:, None]
    disp_sorted = disp[rows, order]
    img_sorted = image_rgb[rows, order]

    for col_idx in range(w):
        u = order[:, col_idx]
        d = disp_sorted[:, col_idx]
        u_r = np.round(u - d).astype(np.int32)
        valid = (u_r >= 0) & (u_r < w)
        for row in np.where(valid)[0]:
            right[row, u_r[row]] = img_sorted[row, col_idx]
            occ_mask[row, u_r[row]] = 0

    if fill_holes:
        right = _fill_holes(right, occ_mask)
    return right, disp


def _fill_holes(image: np.ndarray, hole_mask: np.ndarray) -> np.ndarray:
    """Inpaint occlusion holes with OpenCV's TELEA inpaint."""
    return cv2.inpaint(image, hole_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
