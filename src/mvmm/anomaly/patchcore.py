"""PatchCore — coreset-based patch-feature memory bank.

Reference:
    Roth et al. "Towards Total Recall in Industrial Anomaly Detection."
    CVPR 2022. https://arxiv.org/abs/2106.08265

Why it's still the workhorse in 2026:
    - Training-free in the gradient sense (no backprop). Only forward passes
      through a frozen pretrained backbone + coreset subsampling.
    - Industrially robust — handles slight pose variation through patch-level
      nearest-neighbor matching.
    - Easy to swap backbones (WideResNet50 → DINOv2 → ConvNeXt) and combine
      with classical (rule-based) post-processing.

This implementation aims for clarity & correctness over micro-optimization;
for large memory banks consider FAISS-GPU (see `_NNIndex`).
"""

from __future__ import annotations

import math
import pickle
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from mvmm.anomaly.base import AnomalyDetector, AnomalyResult


# ---------------------------------------------------------------------------
# Feature extractor — torchvision WideResNet50 with hooks on layer2/3.
# ---------------------------------------------------------------------------
class _FeatureExtractor(nn.Module):
    """Extract intermediate features from a frozen WideResNet50."""

    def __init__(self, layers: tuple[str, ...] = ("layer2", "layer3"), device: str = "cuda"):
        super().__init__()
        from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2

        weights = Wide_ResNet50_2_Weights.IMAGENET1K_V2
        self.backbone = wide_resnet50_2(weights=weights)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        self.layers = layers
        self._feats: dict[str, torch.Tensor] = {}
        for name in layers:
            module = dict(self.backbone.named_modules())[name]
            module.register_forward_hook(self._hook(name))

        self.to(device)
        self.device = device

    def _hook(self, name: str):
        def fn(_m, _inp, out):
            self._feats[name] = out

        return fn

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        self._feats.clear()
        _ = self.backbone(x.to(self.device, non_blocking=True))
        # Return a *copy* so subsequent forwards do not overwrite stored feats.
        return {k: v.detach() for k, v in self._feats.items()}


# ---------------------------------------------------------------------------
# Coreset subsampling — greedy farthest-point.
# ---------------------------------------------------------------------------
def _greedy_coreset(features: torch.Tensor, ratio: float, seed: int = 0) -> torch.Tensor:
    """Greedy farthest-point subsampling.

    Args:
        features: (N, D) tensor on any device.
        ratio:    fraction of points to keep, in (0, 1].
    Returns:
        (K, D) tensor of selected samples.
    """
    n = features.shape[0]
    k = max(1, math.ceil(n * ratio))
    if k >= n:
        return features

    gen = torch.Generator(device=features.device).manual_seed(seed)
    start = int(torch.randint(0, n, (1,), generator=gen, device=features.device).item())

    # Min-distance from each candidate to the selected set.
    selected_idx = [start]
    min_dist = torch.cdist(features, features[start : start + 1]).squeeze(1)  # (N,)
    for _ in range(k - 1):
        next_idx = int(torch.argmax(min_dist).item())
        selected_idx.append(next_idx)
        new_dist = torch.cdist(features, features[next_idx : next_idx + 1]).squeeze(1)
        min_dist = torch.minimum(min_dist, new_dist)

    return features[selected_idx]


# ---------------------------------------------------------------------------
# Nearest-neighbor index — pluggable (torch / faiss).
# ---------------------------------------------------------------------------
class _NNIndex:
    """Wraps torch.cdist or FAISS based on availability and bank size."""

    def __init__(self, bank: torch.Tensor, use_faiss: bool = True):
        self.bank_torch = bank
        self.use_faiss = False
        if use_faiss:
            try:
                import faiss  # type: ignore

                d = bank.shape[1]
                index = faiss.IndexFlatL2(d)
                index.add(bank.detach().cpu().numpy().astype(np.float32))
                self._faiss = index
                self.use_faiss = True
            except Exception:
                self.use_faiss = False

    def search(self, queries: torch.Tensor, k: int = 1) -> torch.Tensor:
        """Return (Q,) min-distance per query."""
        if self.use_faiss:
            d, _ = self._faiss.search(queries.detach().cpu().numpy().astype(np.float32), k)
            return torch.from_numpy(d[:, 0]).to(queries.device).sqrt()
        dists = torch.cdist(queries, self.bank_torch)  # (Q, N)
        return dists.min(dim=1).values


# ---------------------------------------------------------------------------
# Helpers: aggregate multi-scale features into a single patch tensor.
# ---------------------------------------------------------------------------
def _aggregate_patches(
    feats: dict[str, torch.Tensor], target_size: int
) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    """Concatenate multi-scale layer outputs after spatial alignment.

    Returns ((B*H*W, D) flattened patches, (B, H, W, D) shape tuple).
    """
    aligned = [F.adaptive_avg_pool2d(v, target_size) for v in feats.values()]
    cat = torch.cat(aligned, dim=1)  # (B, D, H, W)
    b, d, h, w = cat.shape
    flat = cat.permute(0, 2, 3, 1).reshape(b * h * w, d)
    return flat, (b, h, w, d)


def _aggregate_per_image(
    feats: dict[str, torch.Tensor], target_size: int
) -> tuple[torch.Tensor, tuple[int, int, int]]:
    aligned = [F.adaptive_avg_pool2d(v, target_size) for v in feats.values()]
    cat = torch.cat(aligned, dim=1)  # (B, D, H, W)
    b, d, h, w = cat.shape
    return cat.permute(0, 2, 3, 1).reshape(b, h * w, d), (b, h, w)


# ---------------------------------------------------------------------------
# PatchCore main class.
# ---------------------------------------------------------------------------
class PatchCore(AnomalyDetector):
    """PatchCore anomaly detector.

    Args:
        layers:       which backbone stages to use.
        target_size:  spatial resolution to which all feature maps are pooled.
        coreset_ratio: fraction of patches kept after greedy subsampling.
        device:       "cuda" or "cpu".
        use_faiss:    use FAISS for NN search if available.
    """

    name = "patchcore"

    def __init__(
        self,
        layers: tuple[str, ...] = ("layer2", "layer3"),
        target_size: int = 28,
        coreset_ratio: float = 0.1,
        device: str = "cuda",
        use_faiss: bool = True,
    ):
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.target_size = target_size
        self.coreset_ratio = coreset_ratio
        self.use_faiss = use_faiss

        self.extractor = _FeatureExtractor(layers=layers, device=device)
        self._bank: torch.Tensor | None = None
        self._index: _NNIndex | None = None
        self._feat_dim: int | None = None

    # ------------------------------ fit ------------------------------
    def fit(self, loader: Iterable[Any]) -> None:
        all_feats: list[torch.Tensor] = []
        for batch in loader:
            img = batch["image"] if isinstance(batch, dict) else batch
            if not isinstance(img, torch.Tensor):
                raise TypeError("PatchCore expects 'image' to be a torch.Tensor (B,C,H,W)")
            feats = self.extractor(img)
            f, _ = _aggregate_patches(feats, self.target_size)
            all_feats.append(f.cpu())  # CPU staging to free GPU mem
        bank = torch.cat(all_feats, dim=0)
        self._feat_dim = int(bank.shape[1])

        # Greedy coreset on GPU when possible.
        bank_gpu = bank.to(self.device)
        coreset = _greedy_coreset(bank_gpu, ratio=self.coreset_ratio)
        self._bank = coreset.detach()
        self._index = _NNIndex(self._bank, use_faiss=self.use_faiss)

    # ----------------------------- predict ---------------------------
    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> AnomalyResult:
        if self._index is None or self._bank is None:
            raise RuntimeError("PatchCore is not fitted yet; call .fit(loader) first.")
        if images.ndim != 4:
            raise ValueError(f"Expected (B,C,H,W); got shape {tuple(images.shape)}")

        feats = self.extractor(images)
        per_img, (b, h, w) = _aggregate_per_image(feats, self.target_size)  # (B, HW, D)

        score_maps = torch.empty(b, h, w, device=self.device)
        image_scores = torch.empty(b, device=self.device)
        for i in range(b):
            d = self._index.search(per_img[i].to(self.device), k=1)  # (HW,)
            sm = d.view(h, w)
            score_maps[i] = sm
            image_scores[i] = sm.max()

        # Upsample score maps to input resolution.
        sm_up = F.interpolate(
            score_maps.unsqueeze(1), size=images.shape[-2:], mode="bilinear", align_corners=False
        ).squeeze(1)

        return AnomalyResult(
            image_scores=image_scores.detach().cpu().numpy(),
            score_maps=sm_up.detach().cpu().numpy(),
        )

    # ------------------------------ IO -------------------------------
    def save(self, path: str | Path) -> None:
        if self._bank is None:
            raise RuntimeError("Nothing to save — fit first.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "bank": self._bank.detach().cpu().numpy(),
            "feat_dim": self._feat_dim,
            "target_size": self.target_size,
            "coreset_ratio": self.coreset_ratio,
        }
        with path.open("wb") as f:
            pickle.dump(payload, f)

    def load(self, path: str | Path) -> None:
        with Path(path).open("rb") as f:
            payload = pickle.load(f)
        self._bank = torch.from_numpy(payload["bank"]).to(self.device)
        self._feat_dim = payload["feat_dim"]
        self.target_size = payload["target_size"]
        self.coreset_ratio = payload["coreset_ratio"]
        self._index = _NNIndex(self._bank, use_faiss=self.use_faiss)
