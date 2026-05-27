"""SAM2 video predictor wrapper.

Reference:
    Ravi et al. "SAM 2: Segment Anything in Images and Videos." Meta AI, 2024.
    https://github.com/facebookresearch/sam2

SAM2's video predictor maintains a per-track memory of mask propagation
across frames. The typical use is:

    1. Run a *detector* on the first frame.
    2. Convert each box to a SAM2 prompt and seed the memory.
    3. Let SAM2 propagate masks through subsequent frames.

This module is a thin wrapper so the tracking pipeline can call SAM2
without depending on the heavy SAM2 install in tests / CI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class SAM2VideoTracker:
    """Wrap SAM2's video predictor for box-conditioned mask tracking.

    Args:
        checkpoint:  path to a SAM2 .pt checkpoint.
        model_cfg:   model config name (e.g. "sam2_hiera_l.yaml").
        device:      "cuda" or "cpu".
    """

    def __init__(
        self,
        checkpoint: str | Path,
        model_cfg: str = "sam2_hiera_l.yaml",
        device: str = "cuda",
    ):
        try:
            from sam2.build_sam import build_sam2_video_predictor  # type: ignore
        except ImportError as e:
            raise ImportError("sam2 is not installed. See https://github.com/facebookresearch/sam2") from e
        self.device = device
        self.predictor = build_sam2_video_predictor(model_cfg, str(checkpoint), device=device)
        self._state: Any = None

    def init_state(self, video_path: str | Path) -> None:
        """Load a video into SAM2's memory."""
        self._state = self.predictor.init_state(video_path=str(video_path))

    def add_box_prompts(self, frame_idx: int, boxes_xyxy: np.ndarray, obj_ids: list[int]) -> None:
        """Seed SAM2 with one or more box prompts at ``frame_idx``."""
        if self._state is None:
            raise RuntimeError("Call init_state(video_path) first.")
        for box, obj_id in zip(boxes_xyxy, obj_ids, strict=False):
            self.predictor.add_new_points_or_box(
                inference_state=self._state,
                frame_idx=frame_idx,
                obj_id=int(obj_id),
                box=np.asarray(box, dtype=np.float32),
            )

    def propagate(self) -> dict[int, dict[int, np.ndarray]]:
        """Return per-frame, per-object masks.

        Returns:
            dict[frame_idx][obj_id] = HxW uint8 mask.
        """
        if self._state is None:
            raise RuntimeError("Call init_state(video_path) first.")
        out: dict[int, dict[int, np.ndarray]] = {}
        for frame_idx, obj_ids, mask_logits in self.predictor.propagate_in_video(self._state):
            per_frame = {}
            masks = (mask_logits > 0.0).cpu().numpy().astype(np.uint8)
            for i, oid in enumerate(obj_ids):
                per_frame[int(oid)] = masks[i, 0]
            out[int(frame_idx)] = per_frame
        return out
