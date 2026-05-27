"""End-to-end open-vocabulary detect → segment pipeline.

Text prompt ──▶ GroundingDINO/OWLv2 ──▶ boxes ──▶ SAM2 ──▶ masks ──▶ overlay
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mvmm.tracking.detectors import Detections


@dataclass
class OpenVocabResult:
    detections: Detections
    masks: np.ndarray | None  # (N, H, W) uint8 or None if SAM2 not used


class OpenVocabPipeline:
    """Chain a text-conditioned detector with optional SAM2 mask refinement.

    Args:
        detector: GroundingDINODetector or OWLv2Detector (anything with the
                  ``__call__(image_rgb, classes) -> Detections`` signature).
        segmenter: SAM2ImagePredictor or None to skip masking.
    """

    def __init__(self, detector: Any, segmenter: Any | None = None):
        self.detector = detector
        self.segmenter = segmenter

    def __call__(self, image_rgb: np.ndarray, classes: list[str]) -> OpenVocabResult:
        dets = self.detector(image_rgb, classes=classes)
        masks = None
        if self.segmenter is not None and len(dets.boxes):
            seg_out = self.segmenter.predict(image_rgb, boxes_xyxy=dets.boxes)
            masks = seg_out["masks"]
            if masks.ndim == 4 and masks.shape[1] == 1:
                masks = masks[:, 0]
        return OpenVocabResult(detections=dets, masks=masks)
