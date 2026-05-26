"""Synthesize a tiny MVTec-AD-shaped sample dataset for CI / smoke tests.

Creates:
    data/sample/widget/train/good/0..9.png      — clean random textures
    data/sample/widget/test/good/0..1.png       — clean
    data/sample/widget/test/defect/0..2.png     — with a synthetic blob
    data/sample/widget/ground_truth/defect/*    — binary masks

Run:
    python scripts/make_sample_data.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def make_clean(size: int = 256, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.integers(120, 170, size=(size, size, 3), dtype=np.uint8)
    # Add a periodic stripe pattern so it looks part-like.
    y = np.arange(size)
    stripes = (np.sin(y / 6.0) * 12).astype(int).clip(-20, 20)
    base = (base.astype(int) + stripes[:, None, None]).clip(0, 255).astype(np.uint8)
    return base


def add_defect(img: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pil = Image.fromarray(img)
    mask = Image.new("L", pil.size, 0)
    d_img = ImageDraw.Draw(pil)
    d_msk = ImageDraw.Draw(mask)
    h, w = img.shape[:2]
    cx, cy = int(rng.uniform(40, w - 40)), int(rng.uniform(40, h - 40))
    r = int(rng.uniform(8, 22))
    color = (int(rng.uniform(40, 80)),) * 3
    d_img.ellipse((cx - r, cy - r, cx + r, cy + r), fill=color)
    d_msk.ellipse((cx - r, cy - r, cx + r, cy + r), fill=255)
    return np.array(pil), np.array(mask)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/sample/widget"))
    parser.add_argument("--n-train", type=int, default=10)
    parser.add_argument("--n-test-good", type=int, default=2)
    parser.add_argument("--n-test-defect", type=int, default=3)
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()

    root = args.root
    (root / "train" / "good").mkdir(parents=True, exist_ok=True)
    (root / "test" / "good").mkdir(parents=True, exist_ok=True)
    (root / "test" / "defect").mkdir(parents=True, exist_ok=True)
    (root / "ground_truth" / "defect").mkdir(parents=True, exist_ok=True)

    for i in range(args.n_train):
        Image.fromarray(make_clean(args.size, seed=i)).save(root / "train" / "good" / f"{i:03d}.png")
    for i in range(args.n_test_good):
        Image.fromarray(make_clean(args.size, seed=100 + i)).save(root / "test" / "good" / f"{i:03d}.png")
    for i in range(args.n_test_defect):
        clean = make_clean(args.size, seed=200 + i)
        defective, mask = add_defect(clean, seed=300 + i)
        stem = f"{i:03d}"
        Image.fromarray(defective).save(root / "test" / "defect" / f"{stem}.png")
        Image.fromarray(mask).save(root / "ground_truth" / "defect" / f"{stem}_mask.png")

    print(f"Sample dataset created at {root}")


if __name__ == "__main__":
    main()
