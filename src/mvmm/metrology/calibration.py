"""Camera calibration utilities — intrinsics, scale, hand-eye stubs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class CameraIntrinsics:
    """Pinhole intrinsics with distortion coefficients."""

    fx: float
    fy: float
    cx: float
    cy: float
    dist: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    def undistort(self, image: np.ndarray) -> np.ndarray:
        return cv2.undistort(image, self.K, np.array(self.dist, dtype=np.float64))

    @classmethod
    def from_npz(cls, path: str | Path) -> CameraIntrinsics:
        data = np.load(path)
        K = data["K"]
        return cls(
            fx=float(K[0, 0]),
            fy=float(K[1, 1]),
            cx=float(K[0, 2]),
            cy=float(K[1, 2]),
            dist=tuple(map(float, data.get("dist", np.zeros(5)).ravel())),
        )

    def save_npz(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, K=self.K, dist=np.asarray(self.dist, dtype=np.float64))


def calibrate_with_chessboard(
    image_paths: list[str],
    pattern_size: tuple[int, int] = (9, 6),
    square_mm: float = 25.0,
) -> tuple[CameraIntrinsics, float]:
    """Standard OpenCV chessboard calibration.

    Returns the intrinsics and the RMS reprojection error.
    """
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[: pattern_size[0], : pattern_size[1]].T.reshape(-1, 2)
    objp *= square_mm

    obj_points: list[np.ndarray] = []
    img_points: list[np.ndarray] = []
    img_shape: tuple[int, int] | None = None

    for p in image_paths:
        img = cv2.imread(p)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]
        ok, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if not ok:
            continue
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        obj_points.append(objp)
        img_points.append(corners)

    if not obj_points:
        raise RuntimeError("No chessboard corners detected in any image.")

    rms, K, dist, _r, _t = cv2.calibrateCamera(obj_points, img_points, img_shape, None, None)
    return (
        CameraIntrinsics(
            fx=float(K[0, 0]),
            fy=float(K[1, 1]),
            cx=float(K[0, 2]),
            cy=float(K[1, 2]),
            dist=tuple(map(float, dist.ravel())),
        ),
        float(rms),
    )


def pixel_to_mm_scale(known_distance_mm: float, pixel_distance: float) -> float:
    """Closed-form scale factor for planar scenes (known reference target)."""
    if pixel_distance <= 0:
        raise ValueError("pixel_distance must be positive")
    return float(known_distance_mm) / float(pixel_distance)
