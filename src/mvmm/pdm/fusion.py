"""Multimodal PdM model — late fusion of time-series + vision branches.

Outputs:
    - logits over health states (classification), or
    - scalar Remaining Useful Life (regression)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from mvmm.pdm.timeseries import TimesNet
from mvmm.pdm.vision import VisionEncoder


@dataclass
class PdMConfig:
    n_sensor_channels: int
    sensor_seq_len: int = 1024
    sensor_hidden: int = 64
    sensor_depth: int = 2
    vision_backbone: str = "resnet18"
    vision_pretrained: bool = True
    freeze_vision: bool = True
    fusion_dim: int = 128
    num_classes: int | None = 2  # set None for regression
    dropout: float = 0.2


class MultimodalPdMModel(nn.Module):
    """TimesNet (sensor) + Vision (thermal/scope) → fusion MLP → head."""

    def __init__(self, cfg: PdMConfig):
        super().__init__()
        self.cfg = cfg
        self.sensor = TimesNet(
            input_channels=cfg.n_sensor_channels,
            hidden_channels=cfg.sensor_hidden,
            depth=cfg.sensor_depth,
            num_classes=None,
        )
        self.vision = VisionEncoder(
            backbone=cfg.vision_backbone,
            pretrained=cfg.vision_pretrained,
            freeze=cfg.freeze_vision,
        )
        fused_in = cfg.sensor_hidden + self.vision.feat_dim
        self.fuse = nn.Sequential(
            nn.Linear(fused_in, cfg.fusion_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fusion_dim, cfg.fusion_dim),
            nn.GELU(),
        )
        if cfg.num_classes is None:
            self.head = nn.Linear(cfg.fusion_dim, 1)  # RUL regression
        else:
            self.head = nn.Linear(cfg.fusion_dim, cfg.num_classes)

    def forward(self, sensor: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        # sensor: (B, T, C), image: (B, 3, H, W)
        s = self.sensor(sensor)
        v = self.vision(image)
        fused = self.fuse(torch.cat([s, v], dim=-1))
        return self.head(fused)
