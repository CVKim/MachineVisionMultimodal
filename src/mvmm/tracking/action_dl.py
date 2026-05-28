"""Deep action classifier — VideoMAE plug-in for ActivityClassifier.

Reference:
    Tong et al. "VideoMAE: Masked Autoencoders are Data-Efficient
    Learners for Self-Supervised Video Pre-Training." NeurIPS 2022.
    Fine-tuned Kinetics-400 checkpoint:
    ``MCG-NJU/videomae-base-finetuned-kinetics``  (~360 MB, HF Hub)

What this gives you
-------------------
A drop-in replacement for ``ActivityClassifier``'s rule cascade.
Maintains a per-track ring buffer of cropped frames; once the buffer
holds a full clip (16 frames by default), runs VideoMAE on it and
maps the top-1 Kinetics-400 label to our 5-class taxonomy.

Why keep the rule-based path?
-----------------------------
* DL inference cost is non-trivial (~50 ms / clip / track on RTX 3080),
  so we only re-classify every ``every_n_frames`` frames and rely on
  the rule cascade in the gap. The end result is "DL when confident,
  rules elsewhere" — exactly the hybrid the rest of the repo follows.
* The rule path is deterministic + explainable; DL is the upgrade
  for high-stakes deployments.

API mirrors ``ActivityClassifier`` so the rendering / time-accounting
pipeline doesn't need to change.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Mapping from Kinetics-400 labels to our 5-class taxonomy.
# Source: subset of K400 verbs grouped manually by motion semantics.
# Keys are lower-cased substrings — first match wins.
KINETICS_TO_STATE_SUBSTRINGS: tuple[tuple[str, str], ...] = (
    # walking / running family
    ("walking", "walking"),
    ("running", "walking"),
    ("jogging", "walking"),
    ("marching", "walking"),
    ("hiking", "walking"),
    # lifting / picking family
    ("lifting", "lifting"),
    ("carrying", "lifting"),
    ("picking", "lifting"),
    ("pushing", "lifting"),
    ("pulling", "lifting"),
    ("moving furniture", "lifting"),
    ("unloading", "lifting"),
    ("loading", "lifting"),
    # working / manipulation family
    ("assembling", "working"),
    ("sanding", "working"),
    ("welding", "working"),
    ("drilling", "working"),
    ("hammering", "working"),
    ("cutting", "working"),
    ("repairing", "working"),
    ("polishing", "working"),
    ("typing", "working"),
    ("writing", "working"),
    ("cleaning", "working"),
    ("washing", "working"),
    ("cooking", "working"),
    ("painting", "working"),
    ("doing", "working"),
    ("checking", "working"),
    # idle family
    ("sitting", "idle"),
    ("standing", "idle"),
    ("waiting", "idle"),
    ("resting", "idle"),
    ("looking at", "idle"),
    ("reading", "idle"),
    ("watching", "idle"),
    ("staring", "idle"),
)


def map_kinetics_label(label: str) -> str:
    """Map a free-text Kinetics-400 label → our taxonomy."""
    lab = label.lower()
    for keyword, state in KINETICS_TO_STATE_SUBSTRINGS:
        if keyword in lab:
            return state
    return "unknown"


@dataclass
class _DLBuffer:
    """Per-track image-crop ring buffer + cached state."""

    track_id: int
    frames: deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=16))
    last_state: str | None = None
    frames_seen: int = 0


class DeepActionClassifier:
    """VideoMAE-based action classifier with per-track clip buffering.

    Args:
        model_id:        HuggingFace model id for a Kinetics-400 head.
        device:          "cuda" | "cpu".
        clip_len:        number of frames per inference clip.
        every_n_frames:  re-run inference every N new frames per track.
                         Between inferences the cached label is reused.
        crop_pad:        fractional bbox padding before cropping.
    """

    def __init__(
        self,
        model_id: str = "MCG-NJU/videomae-base-finetuned-kinetics",
        device: str = "cuda",
        clip_len: int = 16,
        every_n_frames: int = 8,
        crop_pad: float = 0.15,
    ):
        try:
            import torch
            from transformers import (  # type: ignore
                VideoMAEForVideoClassification,
                VideoMAEImageProcessor,
            )
        except ImportError as e:
            raise ImportError(
                "transformers is required for DeepActionClassifier — `pip install transformers`"
            ) from e
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.clip_len = int(clip_len)
        self.every_n_frames = int(every_n_frames)
        self.crop_pad = float(crop_pad)
        self.model_id = model_id

        self.processor = VideoMAEImageProcessor.from_pretrained(model_id)
        self.model = VideoMAEForVideoClassification.from_pretrained(model_id).to(device).eval()
        # id2label for label mapping.
        self.id2label: dict[int, str] = self.model.config.id2label

        self.buffers: dict[int, _DLBuffer] = {}

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _crop_bbox(frame_rgb: np.ndarray, bbox: np.ndarray, pad: float) -> np.ndarray | None:
        h, w = frame_rgb.shape[:2]
        x1, y1, x2, y2 = (float(v) for v in bbox)
        bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
        x1 = max(int(x1 - pad * bw), 0)
        y1 = max(int(y1 - pad * bh), 0)
        x2 = min(int(x2 + pad * bw), w)
        y2 = min(int(y2 + pad * bh), h)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame_rgb[y1:y2, x1:x2]

    # ----------------------------------------------------------------- update
    def update(self, track_id: int, frame_rgb: np.ndarray, bbox: np.ndarray) -> str | None:
        """Append a frame crop; run inference when the buffer is ready.

        Returns:
            * a state name if we ran an inference this frame
            * the last cached state if the buffer is full but we're skipping
            * ``None`` if the buffer is still warming up
        """
        crop = self._crop_bbox(frame_rgb, bbox, self.crop_pad)
        if crop is None:
            return None
        buf = self.buffers.setdefault(track_id, _DLBuffer(track_id=track_id))
        # VideoMAE wants RGB uint8 frames — keep them as such.
        buf.frames.append(crop.copy())
        buf.frames_seen += 1

        if len(buf.frames) < self.clip_len:
            return None

        if (buf.frames_seen - 1) % max(self.every_n_frames, 1) != 0 and buf.last_state is not None:
            return buf.last_state

        state = self._infer(buf.frames)
        buf.last_state = state
        return state

    # ----------------------------------------------------------------- infer
    def _infer(self, frames: deque[np.ndarray]) -> str:
        import torch

        # VideoMAEImageProcessor expects a *list* of frames per video.
        clip = list(frames)
        inputs = self.processor(clip, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        top = int(logits.argmax(dim=-1).item())
        label = self.id2label.get(top, "")
        return map_kinetics_label(label)

    # ----------------------------------------------------------------- summary
    def summary(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "device": self.device,
            "clip_len": self.clip_len,
            "every_n_frames": self.every_n_frames,
            "active_tracks": len(self.buffers),
        }
