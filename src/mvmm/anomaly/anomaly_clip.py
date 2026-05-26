"""CLIP-based zero-shot / few-shot anomaly detection.

References (2024–2026):
    - WinCLIP: Jeong et al. CVPR 2023. https://arxiv.org/abs/2303.14814
    - AnomalyCLIP: Zhou et al. ICLR 2024. https://arxiv.org/abs/2310.18961
    - FP-CLIP / MultiADS / MuSc-V2 — recent extensions for low FPR.

Why it matters for manufacturing:
    Zero-shot AD lets you screen a new SKU on day one — no normal-only
    training data needed. In a high-mix, low-volume line (semiconductor
    advanced packaging, automotive small-batch parts) this is decisive.

This module implements a compact zero-shot detector along WinCLIP lines:
    - Text prompts describe "good / bad" states of the object.
    - Image is encoded at multiple windows; per-window similarity to the
      "anomalous" text prompt produces a coarse score map.
    - No per-category training; only the CLIP backbone is used.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from mvmm.anomaly.base import AnomalyDetector, AnomalyResult

# Generic, manufacturing-flavored prompt templates.
DEFAULT_NORMAL_PROMPTS = [
    "a photo of a flawless {obj}",
    "a clean {obj} without defect",
    "a perfect {obj}",
    "a {obj} in good condition",
]
DEFAULT_ANOMALY_PROMPTS = [
    "a photo of a {obj} with defect",
    "a damaged {obj}",
    "a {obj} with a scratch",
    "a {obj} with a crack",
    "a {obj} with contamination",
    "a {obj} with a missing part",
]


class AnomalyCLIP(AnomalyDetector):
    """Zero-shot anomaly detection via CLIP vision-language alignment."""

    name = "anomaly-clip"

    def __init__(
        self,
        clip_model: str = "ViT-B-16",
        pretrained: str = "openai",
        object_name: str = "industrial part",
        normal_prompts: list[str] | None = None,
        anomaly_prompts: list[str] | None = None,
        windows: tuple[int, ...] = (2, 3, 4),
        device: str = "cuda",
    ):
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.object_name = object_name
        self.windows = windows

        import open_clip  # type: ignore

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            clip_model, pretrained=pretrained, device=device
        )
        self.tokenizer = open_clip.get_tokenizer(clip_model)
        self.model.eval()

        self.normal_prompts = normal_prompts or [p.format(obj=object_name) for p in DEFAULT_NORMAL_PROMPTS]
        self.anomaly_prompts = anomaly_prompts or [p.format(obj=object_name) for p in DEFAULT_ANOMALY_PROMPTS]

        with torch.no_grad():
            tok_n = self.tokenizer(self.normal_prompts).to(device)
            tok_a = self.tokenizer(self.anomaly_prompts).to(device)
            f_n = self.model.encode_text(tok_n)
            f_a = self.model.encode_text(tok_a)
            self._text_n = F.normalize(f_n.mean(dim=0, keepdim=True), dim=-1)
            self._text_a = F.normalize(f_a.mean(dim=0, keepdim=True), dim=-1)

    def fit(self, _loader: Iterable[Any]) -> None:
        # Zero-shot — no fitting required.
        return None

    @torch.no_grad()
    def _window_scores(self, image: torch.Tensor, n_win: int) -> torch.Tensor:
        """Score by tiling image into n_win x n_win windows and encoding each."""
        b, _c, h, w = image.shape
        win_h, win_w = h // n_win, w // n_win
        scores = torch.zeros(b, n_win, n_win, device=self.device)
        for i in range(n_win):
            for j in range(n_win):
                patch = image[..., i * win_h : (i + 1) * win_h, j * win_w : (j + 1) * win_w]
                patch = F.interpolate(patch, size=(224, 224), mode="bilinear", align_corners=False)
                f = F.normalize(self.model.encode_image(patch), dim=-1)
                sim_n = (f @ self._text_n.T).squeeze(-1)
                sim_a = (f @ self._text_a.T).squeeze(-1)
                # Softmax-like: probability of "anomalous" given the two prompts.
                logits = torch.stack([sim_n, sim_a], dim=-1) * 100.0
                prob_a = F.softmax(logits, dim=-1)[..., 1]
                scores[:, i, j] = prob_a
        return scores

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> AnomalyResult:
        images = images.to(self.device, non_blocking=True)
        b, _c, h, w = images.shape

        # Multi-scale window aggregation — average across scales after upsampling.
        agg = torch.zeros(b, h, w, device=self.device)
        for n in self.windows:
            scores = self._window_scores(images, n)
            up = F.interpolate(
                scores.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False
            ).squeeze(1)
            agg = agg + up
        agg = agg / len(self.windows)
        image_scores = agg.amax(dim=(-1, -2))

        return AnomalyResult(
            image_scores=image_scores.cpu().numpy().astype(np.float32),
            score_maps=agg.cpu().numpy().astype(np.float32),
        )

    # CLIP has no per-fit state; save/load are no-ops for compatibility.
    def save(self, _path: str | Path) -> None:
        return None

    def load(self, _path: str | Path) -> None:
        return None
