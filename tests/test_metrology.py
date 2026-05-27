"""Metrology primitive sanity checks."""

from __future__ import annotations

import numpy as np

from mvmm.three_d.metrology.measure import circle_fit, dimension_from_mask, line_fit
from mvmm.three_d.metrology.segmentation import ClassicalSegmenter


def test_dimension_from_mask_rectangle():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:60, 10:90] = 1  # 40 tall × 80 wide
    dim = dimension_from_mask(mask, scale_mm_per_px=0.5)
    assert abs(min(dim.width_px, dim.height_px) - 40) <= 1
    assert abs(max(dim.width_px, dim.height_px) - 80) <= 1
    assert dim.width_mm is not None and dim.width_mm > 0


def test_circle_fit_disk():
    mask = np.zeros((128, 128), dtype=np.uint8)
    yy, xx = np.mgrid[:128, :128]
    mask[(xx - 64) ** 2 + (yy - 64) ** 2 < 30**2] = 1
    c = circle_fit(mask)
    assert abs(c.cx - 64) < 2 and abs(c.cy - 64) < 2
    assert abs(c.radius_px - 30) < 2


def test_line_fit_diagonal():
    mask = np.zeros((100, 100), dtype=np.uint8)
    for t in range(20, 80):
        mask[t, t] = 1
    line = line_fit(mask)
    # Diagonal length ≈ sqrt(2)*60 ≈ 84.85
    assert 75 < line.length_px < 95


def test_classical_segmenter_box():
    img = (np.random.rand(80, 80, 3) * 255).astype(np.uint8)
    # Paint a distinctive bright square inside
    img[20:60, 20:60] = 240
    seg = ClassicalSegmenter()
    mask = seg(img, box=(15, 15, 50, 50))
    assert mask.dtype == np.uint8 and mask.shape == (80, 80)
    assert mask.sum() > 0
