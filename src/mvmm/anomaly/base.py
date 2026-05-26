"""Abstract base for all anomaly detectors.

Design:
    fit(loader)          — gather normal-distribution statistics
    predict(images)      — return image score + pixel score map
    save / load          — persist memory banks / weights
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class AnomalyResult:
    """Output of an anomaly detector on a batch of images."""

    image_scores: Any  # shape (B,)        float
    score_maps: Any  # shape (B, H, W)   float
    threshold: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "image_scores": self.image_scores,
            "score_maps": self.score_maps,
            "threshold": self.threshold,
        }


class AnomalyDetector(ABC):
    """Abstract anomaly detector interface."""

    name: str = "anomaly-detector"

    @abstractmethod
    def fit(self, loader: Any) -> None:
        """Build the model of normality from a normal-only data loader."""

    @abstractmethod
    def predict(self, images: Any) -> AnomalyResult:
        """Score a batch of images (B, C, H, W) tensors or B-length list."""

    @abstractmethod
    def save(self, path: str | Path) -> None: ...

    @abstractmethod
    def load(self, path: str | Path) -> None: ...
