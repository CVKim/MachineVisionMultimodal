"""Smoke tests for common utilities."""

from __future__ import annotations

import numpy as np

from mvmm.common.metrics import image_auroc, pixel_auroc, pro_score
from mvmm.common.viz import normalize01, overlay_heatmap, side_by_side


def test_image_auroc_perfect_separation():
    scores = np.array([0.1, 0.2, 0.9, 0.95])
    labels = np.array([0, 0, 1, 1])
    assert image_auroc(scores, labels) == 1.0


def test_image_auroc_random_is_half():
    rng = np.random.default_rng(0)
    n = 200
    scores = rng.normal(size=n)
    labels = rng.integers(0, 2, size=n)
    auc = image_auroc(scores, labels)
    assert 0.3 < auc < 0.7


def test_pixel_auroc_perfect():
    sm = np.zeros((1, 16, 16), dtype=np.float32)
    sm[0, 4:12, 4:12] = 1.0
    gt = np.zeros_like(sm, dtype=np.int32)
    gt[0, 4:12, 4:12] = 1
    assert pixel_auroc(sm, gt) == 1.0


def test_pro_score_returns_finite():
    sm = np.zeros((2, 32, 32), dtype=np.float32)
    sm[0, 8:20, 8:20] = 1.0
    sm[1, 12:24, 4:14] = 0.8
    gt = np.zeros_like(sm, dtype=np.int32)
    gt[0, 8:20, 8:20] = 1
    gt[1, 12:24, 4:14] = 1
    val = pro_score(sm, gt)
    assert np.isfinite(val)
    assert 0.0 <= val <= 1.0


def test_normalize01_constant_returns_zero():
    a = np.full((4, 4), 7.0)
    assert (normalize01(a) == 0).all()


def test_overlay_heatmap_shapes():
    img = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    sm = np.random.rand(64, 64).astype(np.float32)
    out = overlay_heatmap(img, sm)
    assert out.shape == img.shape and out.dtype == np.uint8


def test_side_by_side_shapes():
    a = np.zeros((20, 30, 3), dtype=np.uint8)
    b = np.zeros((20, 40, 3), dtype=np.uint8)
    out = side_by_side(a, b, pad=4)
    assert out.shape == (20, 30 + 4 + 40, 3)
