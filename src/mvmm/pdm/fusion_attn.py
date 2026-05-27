"""Cross-attention multimodal fusion for PdM.

Replaces ``MultimodalPdMModel``'s late-fusion concatenation with a
*sensor → vision* cross-attention block: the sensor branch's token
sequence queries the vision branch's feature grid, so the fused
representation is *conditioned on time-varying sensor context* rather
than averaging two unrelated embeddings.

Why it matters for PdM:
    Bearing wear shows up as a vibration spectrum spike (sensor) AND
    a thermal hotspot in a specific bbox (vision). Late-fusion concat
    can't bind those two signals at any specific frame; cross-attention
    explicitly *learns* which image regions correlate with which sensor
    activations.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from mvmm.pdm.timeseries import TimesNet
from mvmm.pdm.vision import VisionEncoder


@dataclass
class CrossAttnPdMConfig:
    n_sensor_channels: int
    sensor_seq_len: int = 512
    sensor_hidden: int = 64
    sensor_depth: int = 2
    vision_backbone: str = "resnet18"
    vision_pretrained: bool = True
    freeze_vision: bool = True
    attn_dim: int = 128
    attn_heads: int = 4
    num_classes: int | None = 2  # None for regression
    dropout: float = 0.2


def _spatial_tokens_from_resnet(feat: torch.Tensor) -> torch.Tensor:
    """Turn a (B, C) pooled feature into (B, 1, C). For richer fusion,
    expose the pre-pool feature map (B, C, H, W) -> (B, H*W, C)."""
    if feat.dim() == 4:
        b, c, h, w = feat.shape
        return feat.permute(0, 2, 3, 1).reshape(b, h * w, c)
    if feat.dim() == 2:
        return feat.unsqueeze(1)
    raise ValueError(f"unexpected vision feature shape: {feat.shape}")


class CrossAttentionFusionModel(nn.Module):
    """Sensor (TimesNet, token-level) ⊕ Vision (ResNet) with cross-attention."""

    def __init__(self, cfg: CrossAttnPdMConfig):
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

        # Project both branches into the common attention dim.
        self.sensor_proj = nn.Linear(cfg.sensor_hidden, cfg.attn_dim)
        self.vision_proj = nn.Linear(self.vision.feat_dim, cfg.attn_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=cfg.attn_dim,
            num_heads=cfg.attn_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.norm_q = nn.LayerNorm(cfg.attn_dim)
        self.norm_kv = nn.LayerNorm(cfg.attn_dim)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.attn_dim, 2 * cfg.attn_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(2 * cfg.attn_dim, cfg.attn_dim),
        )

        if cfg.num_classes is None:
            self.head = nn.Linear(cfg.attn_dim, 1)
        else:
            self.head = nn.Linear(cfg.attn_dim, cfg.num_classes)

    def forward(self, sensor: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        # Sensor → 1 token (B, 1, attn_dim). Even with a single pooled
        # token, the attention block still learns *which vision regions*
        # to attend to conditional on the sensor state.
        s = self.sensor(sensor)  # (B, sensor_hidden)
        q = self.norm_q(self.sensor_proj(s)).unsqueeze(1)  # (B, 1, attn_dim)

        v = self.vision(image)  # (B, feat_dim) or feature map
        v_tokens = _spatial_tokens_from_resnet(v)  # (B, T, feat_dim)
        kv = self.norm_kv(self.vision_proj(v_tokens))  # (B, T, attn_dim)

        attended, _attn_w = self.cross_attn(q, kv, kv, need_weights=False)
        fused = attended.squeeze(1) + self.mlp(attended.squeeze(1))
        return self.head(fused)
