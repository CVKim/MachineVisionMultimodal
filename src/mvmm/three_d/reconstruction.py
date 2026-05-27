"""Scene reconstruction utilities — point cloud from depth, simple meshing.

For full 3D Gaussian Splatting / NeRF, see ``mvmm.three_d.gaussian_splatting``
(wrapper that defers to upstream codebases).
"""

from __future__ import annotations

import numpy as np


def back_project(
    depth_m: np.ndarray,
    intrinsics_3x3: np.ndarray,
    mask: np.ndarray | None = None,
    rgb: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Convert a depth map to an (N, 3) point cloud in camera frame.

    Args:
        depth_m: HxW depth in meters (0 / NaN = invalid).
        intrinsics_3x3: 3x3 camera matrix [[fx,0,cx],[0,fy,cy],[0,0,1]].
        mask: optional HxW boolean — keep only mask>0 pixels.
        rgb: optional HxWx3 uint8 — return matching colors.

    Returns:
        dict with keys "points" (N,3 float32) and optionally "colors" (N,3 uint8).
    """
    h, w = depth_m.shape
    fx, fy = float(intrinsics_3x3[0, 0]), float(intrinsics_3x3[1, 1])
    cx, cy = float(intrinsics_3x3[0, 2]), float(intrinsics_3x3[1, 2])

    yy, xx = np.mgrid[:h, :w]
    valid = np.isfinite(depth_m) & (depth_m > 0)
    if mask is not None:
        valid &= mask.astype(bool)

    z = depth_m[valid].astype(np.float32)
    x = (xx[valid] - cx) * z / fx
    y = (yy[valid] - cy) * z / fy
    points = np.stack([x, y, z], axis=-1)
    out = {"points": points.astype(np.float32)}
    if rgb is not None:
        out["colors"] = rgb[valid].astype(np.uint8)
    return out


def save_ply(path: str, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    """Write a simple ASCII PLY file (no Open3D required)."""
    points = np.asarray(points, dtype=np.float32)
    n = len(points)
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if colors is not None:
        header += ["property uchar red", "property uchar green", "property uchar blue"]
    header.append("end_header")

    lines = ["\n".join(header)]
    if colors is not None:
        colors = np.asarray(colors, dtype=np.uint8)
        for p, c in zip(points, colors, strict=False):
            lines.append(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {int(c[0])} {int(c[1])} {int(c[2])}")
    else:
        for p in points:
            lines.append(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}")
    with open(path, "w", encoding="ascii") as f:
        f.write("\n".join(lines))
