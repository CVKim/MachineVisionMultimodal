"""Video Anomaly Detection (VAD).

Targets surveillance / industrial CCTV anomaly detection: abnormal
events (intrusion, falls, machine stops, dropped objects, etc.).

Modules:
    conv_autoencoder  — frame autoencoder reconstruction-error baseline
    memae             — Memory-Augmented AE (Gong et al. ICCV'19)
    datasets          — UCF-Crime, ShanghaiTech, generic VideoFolder
    train             — training loop (called by scripts/train_vad.py)
"""

from __future__ import annotations

from mvmm.vad.conv_autoencoder import ConvAutoEncoder
from mvmm.vad.datasets import VideoFrameDataset

__all__ = ["ConvAutoEncoder", "VideoFrameDataset"]
