"""3D reconstruction primitives — back-projection + PLY export sanity."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from mvmm.three_d.reconstruction import back_project, save_ply


def test_back_project_returns_n_by_3():
    h, w = 16, 16
    depth = np.full((h, w), 1.5, dtype=np.float32)
    K = np.array([[100.0, 0, 8], [0, 100.0, 8], [0, 0, 1]])
    out = back_project(depth, K)
    assert out["points"].shape == (h * w, 3)
    # All points at z=1.5
    assert np.allclose(out["points"][:, 2], 1.5)


def test_back_project_with_mask_and_rgb():
    h, w = 8, 8
    depth = np.full((h, w), 1.0, dtype=np.float32)
    mask = np.zeros((h, w), dtype=bool)
    mask[2:6, 2:6] = True
    rgb = np.full((h, w, 3), 200, dtype=np.uint8)
    K = np.eye(3)
    K[0, 0] = K[1, 1] = 10
    K[0, 2], K[1, 2] = 4, 4
    out = back_project(depth, K, mask=mask, rgb=rgb)
    assert out["points"].shape == (16, 3)
    assert out["colors"].shape == (16, 3)


def test_save_ply_roundtrip():
    pts = np.array([[0, 0, 0], [1, 2, 3]], dtype=np.float32)
    col = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.uint8)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.ply"
        save_ply(str(path), pts, col)
        body = path.read_text(encoding="ascii")
    assert "ply" in body and "element vertex 2" in body
    assert "0.0000 0.0000 0.0000 10 20 30" in body
