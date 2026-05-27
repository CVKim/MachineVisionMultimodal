"""Depth estimation — monocular (Depth Anything v2) and stereo (SGBM) backends."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


class StereoSGBM:
    """OpenCV SGBM block matching — classical stereo depth.

    Use when you have a well-calibrated stereo rig; produces metric depth
    after multiplying by the rectified baseline and focal length.
    """

    def __init__(
        self,
        num_disparities: int = 96,
        block_size: int = 7,
        focal_px: float = 1000.0,
        baseline_mm: float = 60.0,
    ):
        self.focal = float(focal_px)
        self.baseline = float(baseline_mm)
        self.matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=num_disparities,
            blockSize=block_size,
            P1=8 * 3 * block_size**2,
            P2=32 * 3 * block_size**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
        )

    def __call__(self, left_rgb: np.ndarray, right_rgb: np.ndarray) -> np.ndarray:
        left = cv2.cvtColor(left_rgb, cv2.COLOR_RGB2GRAY)
        right = cv2.cvtColor(right_rgb, cv2.COLOR_RGB2GRAY)
        disp = self.matcher.compute(left, right).astype(np.float32) / 16.0
        # depth_mm = focal * baseline / disparity
        with np.errstate(divide="ignore"):
            depth = (self.focal * self.baseline) / disp
        depth[disp <= 0] = 0
        return depth


class DepthAnythingV2:
    """Lazy wrapper for Depth Anything v2 — best-in-class monocular depth (2024).

    Install:
        pip install transformers
    Weights come from HF Hub automatically.
    """

    def __init__(self, model_id: str = "depth-anything/Depth-Anything-V2-Small-hf", device: str = "cuda"):
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation  # type: ignore
        except ImportError as e:
            raise ImportError("Install transformers: pip install transformers") from e
        import torch

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device).eval()

    def __call__(self, image_rgb: np.ndarray) -> np.ndarray:
        import torch

        inputs = self.processor(images=image_rgb, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        depth = outputs.predicted_depth.squeeze().cpu().numpy().astype(np.float32)
        # Resize to source resolution
        depth = cv2.resize(depth, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        return depth


def build_depth_estimator(backend: str = "stereo_sgbm", **kwargs: Any) -> Any:
    backend = backend.lower()
    if backend == "stereo_sgbm":
        return StereoSGBM(**kwargs)
    if backend in ("depth_anything", "depth_anything_v2"):
        return DepthAnythingV2(**kwargs)
    raise ValueError(f"Unknown depth backend: {backend}")
