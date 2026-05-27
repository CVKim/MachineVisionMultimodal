"""Prompted segmentation wrappers.

We support two backends:
    - SAM2 (Meta, 2024) — best general-purpose prompted segmentation.
    - Classical: GrabCut + edge-refinement fallback for offline / no-GPU.

Both expose the same API:
    seg = build_segmenter(backend="sam2"|"classical", **kwargs)
    mask = seg(image_rgb, points=[(x, y, label), ...])  # or box=(x,y,w,h)
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


class ClassicalSegmenter:
    """GrabCut-based prompted segmentation (no GPU needed)."""

    def __call__(
        self,
        image_rgb: np.ndarray,
        points: list[tuple[int, int, int]] | None = None,
        box: tuple[int, int, int, int] | None = None,
        iterations: int = 5,
    ) -> np.ndarray:
        h, w = image_rgb.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)

        if box is not None:
            mode = cv2.GC_INIT_WITH_RECT
            rect = box
            cv2.grabCut(image_rgb, mask, rect, bgd, fgd, iterations, mode)
        else:
            if not points:
                raise ValueError("Provide at least one point or a box.")
            mode = cv2.GC_INIT_WITH_MASK
            # Seed mask: 1 (probable FG) at FG points, 0 (probable BG) at BG points.
            for x, y, lab in points:
                cv2.circle(mask, (int(x), int(y)), 6, 1 if lab else 0, -1)
            cv2.grabCut(image_rgb, mask, None, bgd, fgd, iterations, mode)
        return ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8)


class SAM2Segmenter:
    """SAM2 wrapper. Lazy-imports the SAM2 package so this stays optional.

    Install: pip install "git+https://github.com/facebookresearch/sam2"
    Weights: download from the SAM2 model card. Pass the local path via
    ``checkpoint``.
    """

    def __init__(self, checkpoint: str, model_cfg: str = "sam2_hiera_l.yaml", device: str = "cuda"):
        try:
            from sam2.build_sam import build_sam2  # type: ignore
            from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore
        except ImportError as e:
            raise ImportError(
                "SAM2 not installed. See https://github.com/facebookresearch/sam2 or "
                "use ClassicalSegmenter as a fallback."
            ) from e
        self.predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint, device=device))

    def __call__(
        self,
        image_rgb: np.ndarray,
        points: list[tuple[int, int, int]] | None = None,
        box: tuple[int, int, int, int] | None = None,
    ) -> np.ndarray:
        self.predictor.set_image(image_rgb)
        pt_coords = pt_labels = None
        if points:
            pt_coords = np.array([[x, y] for x, y, _ in points], dtype=np.float32)
            pt_labels = np.array([lab for _, _, lab in points], dtype=np.int32)
        bbox = None
        if box is not None:
            x, y, w, h = box
            bbox = np.array([x, y, x + w, y + h], dtype=np.float32)
        masks, scores, _ = self.predictor.predict(
            point_coords=pt_coords, point_labels=pt_labels, box=bbox, multimask_output=True
        )
        best = int(np.argmax(scores))
        return masks[best].astype(np.uint8)


def build_segmenter(backend: str = "classical", **kwargs: Any) -> Any:
    backend = backend.lower()
    if backend == "sam2":
        return SAM2Segmenter(**kwargs)
    if backend == "classical":
        return ClassicalSegmenter()
    raise ValueError(f"Unknown segmenter backend: {backend}")
