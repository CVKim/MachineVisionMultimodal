"""Rule-based + deep-learning hybrid anomaly scoring.

Why hybrid wins in real factories:
    - Pure deep models hallucinate on out-of-distribution illumination.
    - Pure rule-based pipelines miss subtle texture/pattern anomalies.
    - Combining them with confidence-weighted fusion gives both
      explainability (rules tell *why*) and recall (DL catches the rest).

Common rule signals in semiconductor / automotive parts inspection:
    - Local LBP / Gabor / GLCM texture deviation
    - Edge-density spikes (scratch, crack)
    - Color/intensity outliers vs reference patch
    - Hough-circle / blob count mismatch against CAD template

This module composes those signals as a single ``RuleBasedScorer`` and
linearly fuses with a DL detector's per-pixel map.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from mvmm.anomaly.base import AnomalyResult


@dataclass
class RuleConfig:
    """Knobs for the classical detector pipeline."""

    lbp_radius: int = 3
    lbp_points: int = 24
    canny_low: int = 50
    canny_high: int = 150
    morph_kernel: int = 3
    intensity_outlier_z: float = 3.0


class RuleBasedScorer:
    """Classical, fast, fully explainable anomaly signals."""

    def __init__(self, config: RuleConfig | None = None):
        self.cfg = config or RuleConfig()

    def edge_density(self, gray: np.ndarray) -> np.ndarray:
        """High-frequency edge density — catches scratches/cracks on smooth surfaces."""
        edges = cv2.Canny(gray, self.cfg.canny_low, self.cfg.canny_high)
        k = self.cfg.morph_kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        density = cv2.dilate(edges, kernel) / 255.0
        return density.astype(np.float32)

    def intensity_outlier(self, gray: np.ndarray) -> np.ndarray:
        """Pixels far from the local mean (z-score) — catches stains, dents."""
        blur = cv2.GaussianBlur(gray, (15, 15), 0).astype(np.float32)
        std = float(gray.std()) + 1e-6
        z = np.abs(gray.astype(np.float32) - blur) / std
        return (z / max(self.cfg.intensity_outlier_z, 1e-6)).clip(0, 1).astype(np.float32)

    def lbp_deviation(self, gray: np.ndarray) -> np.ndarray:
        """LBP histogram deviation vs image-global histogram.

        Implemented in-house (no skimage dependency) for speed.
        """
        r = self.cfg.lbp_radius
        # Approximate LBP using 8-neighbor circular sampling for simplicity.
        h, w = gray.shape
        padded = cv2.copyMakeBorder(gray, r, r, r, r, cv2.BORDER_REFLECT)
        center = padded[r : r + h, r : r + w].astype(np.int32)
        codes = np.zeros((h, w), dtype=np.uint8)
        offsets = [(-r, -r), (-r, 0), (-r, r), (0, r), (r, r), (r, 0), (r, -r), (0, -r)]
        for bit, (dy, dx) in enumerate(offsets):
            shifted = padded[r + dy : r + dy + h, r + dx : r + dx + w].astype(np.int32)
            codes |= (shifted >= center).astype(np.uint8) << bit
        # Per-pixel rarity = 1 - normalized histogram count of its code.
        hist = np.bincount(codes.ravel(), minlength=256).astype(np.float32)
        hist /= hist.sum() + 1e-9
        rarity = 1.0 - hist[codes]
        return rarity.astype(np.float32)

    def score(self, image_rgb: np.ndarray) -> np.ndarray:
        """Return HxW float anomaly score in [0,1]-ish range."""
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        e = self.edge_density(gray)
        o = self.intensity_outlier(gray)
        l_ = self.lbp_deviation(gray)
        # Equal-weight ensemble — domain teams typically retune these per part.
        return (0.4 * e + 0.3 * o + 0.3 * l_).astype(np.float32)


def fuse_rule_and_dl(
    rule_map: np.ndarray,
    dl_result: AnomalyResult,
    alpha: float = 0.5,
) -> AnomalyResult:
    """Linear fusion of a rule map with a DL detector's result.

    Args:
        rule_map:  HxW or (N,H,W) score from RuleBasedScorer.
        dl_result: output of any AnomalyDetector.
        alpha:     weight of the DL branch in [0,1].
    """
    rm = np.asarray(rule_map, dtype=np.float32)
    dl_map = np.asarray(dl_result.score_maps, dtype=np.float32)
    if rm.ndim == 2:
        rm = rm[None, ...]
    # Min-max normalize each separately so weights compose meaningfully.
    rm_n = _norm(rm)
    dl_n = _norm(dl_map)
    fused = alpha * dl_n + (1.0 - alpha) * rm_n
    image_scores = fused.reshape(fused.shape[0], -1).max(axis=1)
    return AnomalyResult(image_scores=image_scores, score_maps=fused)


def _norm(a: np.ndarray) -> np.ndarray:
    flat = a.reshape(a.shape[0], -1)
    lo = flat.min(axis=1, keepdims=True)
    hi = flat.max(axis=1, keepdims=True)
    out = (flat - lo) / (hi - lo + 1e-9)
    return out.reshape(a.shape).astype(np.float32)
