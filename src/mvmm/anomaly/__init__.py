"""Image-level anomaly / defect-detection algorithms.

For video anomaly detection (CCTV / surveillance), see `mvmm.vad`.
For zero-shot anomaly with vision-language models, see `mvmm.zeroshot.anomaly_clip`.

Module map:
    base          — abstract base class for all image-AD detectors
    patchcore     — PatchCore (Roth et al. CVPR 2022) — memory-bank baseline
    efficient_ad  — EfficientAD (Batzner et al. WACV 2024) — real-time S/T+AE
    hybrid        — rule-based + DL fusion for high-precision manufacturing
"""

from __future__ import annotations

from mvmm.anomaly.base import AnomalyDetector, AnomalyResult
from mvmm.anomaly.patchcore import PatchCore

__all__ = ["AnomalyDetector", "AnomalyResult", "PatchCore"]
