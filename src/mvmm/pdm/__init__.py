"""Predictive Maintenance (PdM) — multimodal sensor + vision fusion.

Modules:
    timeseries     — TimesNet-style 1D backbone for vibration / current signals
    vision         — image feature extractor (thermal / scope camera frames)
    fusion         — late-fusion classifier / RUL regressor over both branches
    datasets       — generic CSV/parquet + image-folder PdM dataset

Why multimodal for PdM:
    A single sensor channel often misses cross-modal signatures of
    impending failure (e.g. bearing wear shows up *together* in vibration
    spectrum and infrared heat signature). Fusing both lifts recall
    without inflating false alarms.
"""

from __future__ import annotations

from mvmm.pdm.fusion import MultimodalPdMModel
from mvmm.pdm.fusion_attn import CrossAttentionFusionModel, CrossAttnPdMConfig
from mvmm.pdm.patchtst import PatchTST
from mvmm.pdm.timeseries import TimesNet, TimesNetBlock

__all__ = [
    "CrossAttentionFusionModel",
    "CrossAttnPdMConfig",
    "MultimodalPdMModel",
    "PatchTST",
    "TimesNet",
    "TimesNetBlock",
]
