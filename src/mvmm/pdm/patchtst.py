"""PatchTST — channel-independent transformer for multivariate time series.

Reference:
    Nie et al. "A Time Series is Worth 64 Words: Long-term Forecasting
    with Transformers." ICLR 2023. https://arxiv.org/abs/2211.14730

Two design choices distinguish it from TimesNet:

    * **Channel independence** — each sensor channel is encoded *separately*
      through the transformer, then concatenated at the head. This dodges
      the curse of high-dimensional input and is empirically strong on
      noisy industrial signals.
    * **Patching** — the per-channel sequence is split into non-overlapping
      L-length patches before the transformer, dramatically reducing the
      token count and making long sequences tractable.

This is a clean PyTorch port suitable as a drop-in alternative to
``mvmm.pdm.timeseries.TimesNet``.
"""

from __future__ import annotations

import torch
from torch import nn


class _ChannelIndependentEncoder(nn.Module):
    """Encode one channel's patches through a small transformer."""

    def __init__(self, patch_len: int, d_model: int, n_heads: int, n_layers: int, dropout: float):
        super().__init__()
        self.proj_in = nn.Linear(patch_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_patches, patch_len)  ->  (B, d_model) pooled."""
        h = self.proj_in(x)
        h = self.encoder(h)
        h = self.norm(h)
        return h.mean(dim=1)


class PatchTST(nn.Module):
    """Channel-independent PatchTST.

    Args:
        n_channels:    number of input sensor channels.
        seq_len:       input sequence length.
        patch_len:     non-overlapping patch length (must divide seq_len).
        d_model:       transformer hidden dim.
        n_heads:       attention heads.
        n_layers:      transformer encoder layers.
        num_classes:   set to None for regression (RUL); else classification.
        dropout:       residual dropout rate.
    """

    def __init__(
        self,
        n_channels: int,
        seq_len: int = 512,
        patch_len: int = 32,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        num_classes: int | None = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if seq_len % patch_len != 0:
            raise ValueError(f"seq_len ({seq_len}) must be divisible by patch_len ({patch_len})")
        self.n_channels = int(n_channels)
        self.seq_len = int(seq_len)
        self.patch_len = int(patch_len)
        self.n_patches = seq_len // patch_len
        self.d_model = int(d_model)

        # Shared encoder across channels (channel-independent processing).
        self.encoder = _ChannelIndependentEncoder(
            patch_len=self.patch_len,
            d_model=self.d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )

        head_in = self.n_channels * self.d_model
        if num_classes is None:
            self.head = nn.Linear(head_in, 1)  # RUL regression
        else:
            self.head = nn.Linear(head_in, int(num_classes))
        self.feat_dim = head_in

    def encode_features(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C)  ->  (B, C*d_model) concatenated per-channel embedding."""
        b, t, c = x.shape
        if t != self.seq_len or c != self.n_channels:
            raise ValueError(f"Expected (*, {self.seq_len}, {self.n_channels}); got (*, {t}, {c})")
        # (B, T, C) -> (B, C, T) -> patches: (B, C, n_patches, patch_len)
        patches = x.permute(0, 2, 1).reshape(b, c, self.n_patches, self.patch_len)
        # Encode each channel through the shared encoder.
        feats: list[torch.Tensor] = []
        for ch in range(c):
            feats.append(self.encoder(patches[:, ch]))  # (B, d_model)
        return torch.cat(feats, dim=-1)  # (B, C * d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode_features(x))
