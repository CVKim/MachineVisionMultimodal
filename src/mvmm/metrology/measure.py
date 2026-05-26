"""Primitive measurement from masks: lengths, circles, diameters.

These are the classic tools manufacturing metrology has relied on for
years; we keep them deliberately simple and dependency-light, so they
can be combined with any segmentation source (SAM2, GrabCut, classical
edge detection).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CircleFit:
    cx: float
    cy: float
    radius_px: float
    radius_mm: float | None = None


@dataclass
class LineFit:
    x1: float
    y1: float
    x2: float
    y2: float
    length_px: float
    length_mm: float | None = None


@dataclass
class DimensionResult:
    width_px: float
    height_px: float
    width_mm: float | None = None
    height_mm: float | None = None
    rotated_box: np.ndarray | None = None  # 4x2


def _largest_contour(mask: np.ndarray) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No contour found in mask.")
    return max(contours, key=cv2.contourArea)


def circle_fit(mask: np.ndarray, scale_mm_per_px: float | None = None) -> CircleFit:
    """Fit a minimum-enclosing circle around the largest mask region."""
    cnt = _largest_contour(mask)
    (cx, cy), r = cv2.minEnclosingCircle(cnt)
    return CircleFit(
        cx=float(cx),
        cy=float(cy),
        radius_px=float(r),
        radius_mm=float(r) * scale_mm_per_px if scale_mm_per_px else None,
    )


def line_fit(mask: np.ndarray, scale_mm_per_px: float | None = None) -> LineFit:
    """Fit a line through the mask using cv2.fitLine; report endpoints by projection."""
    cnt = _largest_contour(mask)
    [vx, vy, x0, y0] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    proj = (cnt[:, 0, 0] - x0) * vx + (cnt[:, 0, 1] - y0) * vy
    t_min, t_max = float(proj.min()), float(proj.max())
    x1, y1 = x0 + t_min * vx, y0 + t_min * vy
    x2, y2 = x0 + t_max * vx, y0 + t_max * vy
    length_px = float(np.hypot(x2 - x1, y2 - y1))
    return LineFit(
        x1=float(x1),
        y1=float(y1),
        x2=float(x2),
        y2=float(y2),
        length_px=length_px,
        length_mm=length_px * scale_mm_per_px if scale_mm_per_px else None,
    )


def dimension_from_mask(mask: np.ndarray, scale_mm_per_px: float | None = None) -> DimensionResult:
    """Min-area rotated rectangle → width/height (px and optionally mm)."""
    cnt = _largest_contour(mask)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    (w_px, h_px) = sorted(rect[1])  # ensure w<=h consistently
    return DimensionResult(
        width_px=float(w_px),
        height_px=float(h_px),
        width_mm=float(w_px) * scale_mm_per_px if scale_mm_per_px else None,
        height_mm=float(h_px) * scale_mm_per_px if scale_mm_per_px else None,
        rotated_box=box,
    )
