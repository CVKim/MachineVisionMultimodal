"""Visualization helpers — anomaly heatmaps, bounding boxes, side-by-side panels."""

from __future__ import annotations

import numpy as np


def normalize01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def overlay_heatmap(
    image_rgb: np.ndarray,
    score_map: np.ndarray,
    alpha: float = 0.45,
    colormap: str = "jet",
) -> np.ndarray:
    """Overlay a score map on an RGB image.

    Args:
        image_rgb: HxWx3 uint8.
        score_map: HxW float (any range — will be normalized to [0,1]).
        alpha:     blend factor for the heatmap.
    Returns:
        HxWx3 uint8 array.
    """
    import matplotlib

    cmap = matplotlib.colormaps[colormap]
    norm = normalize01(score_map)
    heat = (cmap(norm)[..., :3] * 255).astype(np.uint8)

    if image_rgb.shape[:2] != heat.shape[:2]:
        # Resize the score map to match the image
        from PIL import Image

        heat_img = Image.fromarray(heat).resize((image_rgb.shape[1], image_rgb.shape[0]), Image.BILINEAR)
        heat = np.asarray(heat_img)

    blended = alpha * heat.astype(np.float32) + (1 - alpha) * image_rgb.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


def side_by_side(*panels: np.ndarray, pad: int = 8, bg: int = 255) -> np.ndarray:
    """Horizontal montage with a uniform pad gap."""
    h = max(p.shape[0] for p in panels)
    total_w = sum(p.shape[1] for p in panels) + pad * (len(panels) - 1)
    canvas = np.full((h, total_w, 3), bg, dtype=np.uint8)
    x = 0
    for p in panels:
        if p.ndim == 2:
            p = np.stack([p] * 3, axis=-1)
        canvas[: p.shape[0], x : x + p.shape[1]] = p
        x += p.shape[1] + pad
    return canvas
