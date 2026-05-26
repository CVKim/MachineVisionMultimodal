"""Anomaly / defect-detection algorithms.

Module map:
    base          — abstract base class for all detectors
    patchcore     — PatchCore (Roth et al. CVPR 2022) — memory-bank baseline
    efficient_ad  — EfficientAD (Batzner et al. WACV 2024) — real-time S/T+AE
    anomaly_clip  — CLIP-based zero-shot AD (WinCLIP / AnomalyCLIP-style)
    hybrid        — rule-based + DL fusion for high-precision manufacturing
"""

from __future__ import annotations

from mvmm.anomaly.base import AnomalyDetector, AnomalyResult
from mvmm.anomaly.patchcore import PatchCore

__all__ = ["AnomalyDetector", "AnomalyResult", "PatchCore"]
