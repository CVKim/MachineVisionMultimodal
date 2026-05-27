"""Zero-shot / open-vocabulary perception via foundation models.

Vision-language models (CLIP, GroundingDINO, OWLv2, SAM2) let you
detect / segment / score novel objects with **no labeled training
data** — only text prompts. This is decisive on high-mix lines where
new SKUs appear faster than annotation can keep up.

Modules:
    anomaly_clip       — WinCLIP/AnomalyCLIP-style image AD
    grounding_dino     — open-vocabulary detection ("text → boxes")
    owl_v2             — OWLv2 zero-shot detection (HF transformers)
    sam2_promptable    — SAM2 image predictor with point/box/text prompts
    pipeline           — chain: text prompt → boxes → masks → scores
"""

from __future__ import annotations

from mvmm.zeroshot.anomaly_clip import AnomalyCLIP
from mvmm.zeroshot.grounding_dino import GroundingDINODetector
from mvmm.zeroshot.owl_v2 import OWLv2Detector
from mvmm.zeroshot.pipeline import OpenVocabPipeline, OpenVocabResult
from mvmm.zeroshot.sam2_promptable import SAM2ImagePredictor

__all__ = [
    "AnomalyCLIP",
    "GroundingDINODetector",
    "OWLv2Detector",
    "OpenVocabPipeline",
    "OpenVocabResult",
    "SAM2ImagePredictor",
]
