"""SAM2 image predictor (single-frame, promptable segmentation).

For video propagation see ``mvmm.tracking.sam2_video``.

Pipeline pattern:
    1. detector (GroundingDINO / OWLv2) gives boxes from a text prompt
    2. SAM2 image predictor refines each box into a pixel-accurate mask

This is the open-vocabulary equivalent of "click to segment" — but the
click is replaced by a text phrase.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class SAM2ImagePredictor:
    """Thin wrapper around SAM2's single-image predictor.

    Args:
        checkpoint: path to a SAM2 .pt checkpoint.
        model_cfg:  Hydra config name (e.g. "sam2_hiera_l.yaml").
        device:     "cuda" or "cpu".
    """

    def __init__(self, checkpoint: str | Path, model_cfg: str = "sam2_hiera_l.yaml", device: str = "cuda"):
        try:
            from sam2.build_sam import build_sam2  # type: ignore
            from sam2.sam2_image_predictor import SAM2ImagePredictor as _Impl  # type: ignore
        except ImportError as e:
            raise ImportError("sam2 is not installed. See https://github.com/facebookresearch/sam2") from e
        self.device = device
        self.predictor = _Impl(build_sam2(model_cfg, str(checkpoint), device=device))

    def predict(
        self,
        image_rgb: np.ndarray,
        boxes_xyxy: np.ndarray | None = None,
        points: list[tuple[int, int, int]] | None = None,
    ) -> dict[str, Any]:
        """Run SAM2 on an image, optionally conditioned on boxes or points."""
        self.predictor.set_image(image_rgb)
        if boxes_xyxy is not None and len(boxes_xyxy):
            masks, scores, _ = self.predictor.predict(
                box=boxes_xyxy.astype(np.float32),
                multimask_output=False,
            )
        elif points:
            pt_coords = np.array([[x, y] for x, y, _ in points], dtype=np.float32)
            pt_labels = np.array([lab for _, _, lab in points], dtype=np.int32)
            masks, scores, _ = self.predictor.predict(
                point_coords=pt_coords,
                point_labels=pt_labels,
                multimask_output=True,
            )
        else:
            raise ValueError("Provide at least one of boxes_xyxy or points.")
        return {"masks": np.asarray(masks).astype(np.uint8), "scores": np.asarray(scores)}
