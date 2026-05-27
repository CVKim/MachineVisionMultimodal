"""Memory-Augmented AutoEncoder for VAD.

Reference:
    Gong et al. "Memorizing Normality to Detect Anomaly: Memory-Augmented
    Deep Autoencoder for Unsupervised Anomaly Detection." ICCV 2019.
    https://arxiv.org/abs/1904.02639

Idea:
    A plain AE generalizes too well — it can reconstruct anomalies
    almost as well as normals, which collapses the score gap. MemAE
    inserts a *memory bank* of learnable prototypes between encoder
    and decoder. The bottleneck is forced to be a convex combination
    of the prototypes, so anomalies (far from any prototype) suffer
    bigger reconstruction error.

This is the canonical strong baseline for surveillance VAD — works
well on ShanghaiTech, UCSDped2, Avenue.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from mvmm.vad.conv_autoencoder import _conv_block, _deconv_block


class MemoryModule(nn.Module):
    """Bank of N learnable prototypes addressed by cosine similarity."""

    def __init__(self, n_slots: int, dim: int, shrink_thresh: float = 1.0 / 500.0):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(n_slots, dim) * 0.02)
        self.shrink_thresh = float(shrink_thresh)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """z: (B, C, H, W)  →  reconstructed z, attention weights."""
        b, c, h, w = z.shape
        flat = z.permute(0, 2, 3, 1).reshape(-1, c)  # (BHW, C)
        flat_n = F.normalize(flat, dim=-1)
        mem_n = F.normalize(self.memory, dim=-1)
        att = torch.softmax(flat_n @ mem_n.T, dim=-1)  # (BHW, N)
        # Hard-shrink: zero out attention below threshold (sharpens recall).
        att = (att - self.shrink_thresh).clamp(min=0)
        att = att / (att.sum(dim=-1, keepdim=True) + 1e-9)
        out_flat = att @ self.memory  # (BHW, C)
        out = out_flat.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        return out, att


class MemAE(nn.Module):
    """Memory-augmented frame autoencoder."""

    def __init__(self, in_channels: int = 3, base: int = 32, latent: int = 256, n_slots: int = 1000):
        super().__init__()
        self.encoder = nn.Sequential(
            _conv_block(in_channels, base, 2),
            _conv_block(base, base * 2, 2),
            _conv_block(base * 2, base * 4, 2),
            _conv_block(base * 4, latent, 2),
        )
        self.memory = MemoryModule(n_slots=n_slots, dim=latent)
        self.decoder = nn.Sequential(
            _deconv_block(latent, base * 4, 2),
            _deconv_block(base * 4, base * 2, 2),
            _deconv_block(base * 2, base, 2),
            nn.ConvTranspose2d(base, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        z_mem, att = self.memory(z)
        recon = self.decoder(z_mem)
        return recon, att

    @torch.no_grad()
    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        recon, _ = self(x)
        return (recon - x).pow(2).mean(dim=(1, 2, 3))


def entropy_loss(att: torch.Tensor) -> torch.Tensor:
    """Encourage sparse attention over the memory bank."""
    return (-att * torch.log(att + 1e-12)).sum(dim=-1).mean()
