"""TimesNet-style backbone for multivariate time series.

Reference:
    Wu et al. "TimesNet: Temporal 2D-Variation Modeling for General Time
    Series Analysis." ICLR 2023. https://arxiv.org/abs/2210.02186

Why it fits PdM:
    Reshaping the 1D signal into a 2D periodic image lets us reuse strong
    2D CNN architectures (Inception/ConvNeXt) — sees both intra-period
    and inter-period variation in one shot. Strong for vibration spectra
    and motor current signatures.

We provide a compact ``TimesNetBlock`` that you can stack into a model.
Full top-level model lives in ``fusion.py``.
"""

from __future__ import annotations

import torch
import torch.fft
import torch.nn.functional as F
from torch import nn


def _find_top_k_periods(signal: torch.Tensor, k: int) -> torch.Tensor:
    """Estimate top-k periods of a (B, T, C) signal via FFT magnitude.

    Returns (k,) period lengths (ints) — shared across the batch.
    """
    xf = torch.fft.rfft(signal, dim=1)
    amp = xf.abs().mean(dim=(0, 2))  # (T//2 + 1,)
    amp[0] = 0  # ignore DC
    _, idx = torch.topk(amp, k)
    # Period length = T / freq_index. Avoid div-by-zero (excluded above).
    periods = signal.shape[1] // idx.clamp(min=1)
    return periods


class _InceptionBlock(nn.Module):
    """Tiny inception-style 2D conv block used inside TimesNet."""

    def __init__(self, channels: int):
        super().__init__()
        self.b1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.b3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.b5 = nn.Conv2d(channels, channels, kernel_size=5, padding=2)
        self.fuse = nn.Conv2d(channels * 3, channels, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.cat([self.b1(x), self.b3(x), self.b5(x)], dim=1)
        return self.act(self.fuse(y))


class TimesNetBlock(nn.Module):
    """One TimesNet block: top-k periods → reshape to 2D → inception → aggregate.

    Args:
        channels:  hidden channel size (must equal input feature dim).
        top_k:     number of periods to consider per block.
    """

    def __init__(self, channels: int, top_k: int = 3):
        super().__init__()
        self.top_k = top_k
        self.inception = _InceptionBlock(channels)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        b, t, c = x.shape
        periods = _find_top_k_periods(x, self.top_k)

        outs = []
        weights = []
        for p in periods.tolist():
            p = max(int(p), 1)
            # Pad T up to multiple of p
            t_pad = ((t + p - 1) // p) * p
            pad_amt = t_pad - t
            xp = F.pad(x, (0, 0, 0, pad_amt))
            # Reshape to (B, C, t_pad/p, p) = (B, C, n_periods, period)
            xp = xp.permute(0, 2, 1).contiguous().reshape(b, c, t_pad // p, p)
            xp = self.inception(xp)
            xp = xp.reshape(b, c, t_pad).permute(0, 2, 1)[:, :t, :]
            outs.append(xp)
            # Use amplitude at the period as adaptive weight.
            with torch.no_grad():
                xf = torch.fft.rfft(x, dim=1).abs().mean(dim=(0, 2))
                idx = min(t // max(p, 1), xf.shape[0] - 1)
                weights.append(xf[idx].clamp(min=1e-6))
        weights_t = torch.softmax(torch.stack(weights), dim=0)
        stacked = torch.stack(outs, dim=0)  # (k, B, T, C)
        y = (stacked * weights_t.view(-1, 1, 1, 1)).sum(dim=0)
        return self.norm(x + y)


class TimesNet(nn.Module):
    """Stack of TimesNetBlocks → mean-pool → linear head."""

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int = 64,
        depth: int = 2,
        num_classes: int | None = None,
        top_k: int = 3,
    ):
        super().__init__()
        self.proj_in = nn.Linear(input_channels, hidden_channels)
        self.blocks = nn.ModuleList([TimesNetBlock(hidden_channels, top_k=top_k) for _ in range(depth)])
        self.head: nn.Module | None
        if num_classes is not None:
            self.head = nn.Linear(hidden_channels, num_classes)
        else:
            self.head = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, input_channels)
        h = self.proj_in(x)
        for blk in self.blocks:
            h = blk(h)
        pooled = h.mean(dim=1)  # (B, hidden)
        if self.head is None:
            return pooled
        return self.head(pooled)
