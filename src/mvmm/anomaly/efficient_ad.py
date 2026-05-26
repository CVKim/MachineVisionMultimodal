"""EfficientAD — student/teacher + autoencoder for real-time anomaly detection.

Reference:
    Batzner et al. "EfficientAD: Accurate Visual Anomaly Detection at
    Millisecond-level Latencies." WACV 2024.
    https://arxiv.org/abs/2303.14535

Design notes:
    - Teacher: PDN (Patch Description Network) pretrained on ImageNet.
    - Student: same PDN architecture, trained to mimic the teacher on
      normal images and to be confused on randomly-sampled distractor
      images (hard-feature loss).
    - Autoencoder branch detects logical anomalies that S-T can miss.
    - Anomaly map = max(student-teacher map, autoencoder map).

This module declares the network architectures and the training loop signature.
The full training pipeline can be wired up in scripts/train_efficient_ad.py;
for the first iteration we keep weights trainable and the PDN definition
faithful to the paper's "S" variant.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from mvmm.anomaly.base import AnomalyDetector, AnomalyResult


def _conv_block(in_c: int, out_c: int, k: int = 4, s: int = 1, p: int = 0) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p),
        nn.ReLU(inplace=True),
    )


class PDN_Small(nn.Module):
    """Patch Description Network — "small" variant from the paper."""

    def __init__(self, out_channels: int = 384):
        super().__init__()
        self.net = nn.Sequential(
            _conv_block(3, 128, k=4, s=1, p=3),
            nn.AvgPool2d(kernel_size=2, stride=2),
            _conv_block(128, 256, k=4, s=1, p=3),
            nn.AvgPool2d(kernel_size=2, stride=2),
            _conv_block(256, 256, k=3, s=1, p=1),
            nn.Conv2d(256, out_channels, kernel_size=4, stride=1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AutoEncoder(nn.Module):
    """Lightweight AE for the logical-anomaly branch."""

    def __init__(self, out_channels: int = 384):
        super().__init__()
        self.encoder = nn.Sequential(
            _conv_block(3, 32, 4, 2, 1),
            _conv_block(32, 64, 4, 2, 1),
            _conv_block(64, 64, 4, 2, 1),
            _conv_block(64, 64, 4, 2, 1),
            nn.Conv2d(64, 64, 4, 1, 0),
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            _conv_block(64, 64, 3, 1, 1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            _conv_block(64, 64, 3, 1, 1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            _conv_block(64, 64, 3, 1, 1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, out_channels, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


class EfficientAD(AnomalyDetector):
    """EfficientAD (small variant). Real-time, MVTec-AD strong baseline."""

    name = "efficient-ad"

    def __init__(self, device: str = "cuda", channels: int = 384):
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.channels = channels
        self.teacher = PDN_Small(channels).to(device).eval()
        self.student = PDN_Small(2 * channels).to(device).train()
        self.autoencoder = AutoEncoder(channels).to(device).train()
        # Normalization stats — populated during fit.
        self._teacher_mean: torch.Tensor | None = None
        self._teacher_std: torch.Tensor | None = None

    def fit(self, loader: Any) -> None:
        # Training loop is intentionally implemented in scripts/train_efficient_ad.py
        # to keep the model class focused. See docs/ROADMAP.md for status.
        raise NotImplementedError(
            "EfficientAD training is wired in scripts/train_efficient_ad.py "
            "(see ROADMAP.md). The class itself defines the network only for now."
        )

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> AnomalyResult:
        images = images.to(self.device, non_blocking=True)
        t = self.teacher(images)
        s = self.student(images)
        s_st = s[:, : self.channels]
        s_ae = s[:, self.channels :]
        ae = self.autoencoder(images)

        # Student-teacher map.
        st_map = (t - s_st).pow(2).mean(dim=1)
        # Autoencoder map.
        ae_map = (ae - s_ae).pow(2).mean(dim=1)
        anomaly_map = torch.maximum(st_map, ae_map)

        anomaly_map_up = F.interpolate(
            anomaly_map.unsqueeze(1),
            size=images.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        image_scores = anomaly_map_up.amax(dim=(-1, -2))

        return AnomalyResult(
            image_scores=image_scores.detach().cpu().numpy(),
            score_maps=anomaly_map_up.detach().cpu().numpy(),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "teacher": self.teacher.state_dict(),
                "student": self.student.state_dict(),
                "autoencoder": self.autoencoder.state_dict(),
                "channels": self.channels,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.teacher.load_state_dict(ckpt["teacher"])
        self.student.load_state_dict(ckpt["student"])
        self.autoencoder.load_state_dict(ckpt["autoencoder"])
        self.channels = ckpt["channels"]
