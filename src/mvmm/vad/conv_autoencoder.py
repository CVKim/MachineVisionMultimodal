"""Frame autoencoder baseline for Video Anomaly Detection.

The simplest, most reproducible VAD baseline that still trains in minutes:

    encoder: 4-stage 2D CNN downsampling (H/16, W/16, C=64)
    decoder: 4-stage transposed conv upsampling
    train:   minimize MSE reconstruction error on *normal* clips
    test:    anomaly score = per-frame reconstruction error
             (high error = unfamiliar = anomalous)

This is the basis for the modern MemAE / MGFN line of work — those
swap the bottleneck with a learnable memory bank to harden the gap
between normal and anomalous reconstructions. We expose the plain AE
here so the training loop is clear; ``memae.py`` adds the memory step.
"""

from __future__ import annotations

import torch
from torch import nn


def _conv_block(in_c: int, out_c: int, stride: int = 2) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1),
        nn.BatchNorm2d(out_c),
        nn.LeakyReLU(0.2, inplace=True),
    )


def _deconv_block(in_c: int, out_c: int, stride: int = 2) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=stride, padding=1),
        nn.BatchNorm2d(out_c),
        nn.LeakyReLU(0.2, inplace=True),
    )


class ConvAutoEncoder(nn.Module):
    """Encoder → bottleneck → decoder with a single-frame input.

    Args:
        in_channels: 3 for RGB, 1 for grayscale.
        base: width of the first conv (multiplied by 2 each stage).
        latent: bottleneck channels.
    """

    def __init__(self, in_channels: int = 3, base: int = 32, latent: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            _conv_block(in_channels, base, 2),
            _conv_block(base, base * 2, 2),
            _conv_block(base * 2, base * 4, 2),
            _conv_block(base * 4, latent, 2),
        )
        self.decoder = nn.Sequential(
            _deconv_block(latent, base * 4, 2),
            _deconv_block(base * 4, base * 2, 2),
            _deconv_block(base * 2, base, 2),
            nn.ConvTranspose2d(base, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)

    @torch.no_grad()
    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Per-image anomaly score = mean(per-pixel squared error)."""
        recon = self(x)
        err = (recon - x).pow(2).mean(dim=(1, 2, 3))
        return err
