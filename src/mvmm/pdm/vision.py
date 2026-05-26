"""Image branch for the PdM model — thermal / scope-cam frame encoder."""

from __future__ import annotations

import torch
from torch import nn


class VisionEncoder(nn.Module):
    """Frozen torchvision backbone → global pooled feature.

    By default uses ResNet18 (cheap, fast). Swap to timm models for stronger
    features on small thermal datasets (e.g. ``timm:efficientnet_b0``).
    """

    def __init__(self, backbone: str = "resnet18", pretrained: bool = True, freeze: bool = True):
        super().__init__()
        if backbone == "resnet18":
            from torchvision.models import ResNet18_Weights, resnet18

            net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            net.fc = nn.Identity()
            self.feat_dim = 512
            self.backbone = net
        else:
            import timm  # type: ignore

            self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
            self.feat_dim = self.backbone.num_features

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            self.backbone.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training and isinstance(self.backbone, nn.Module):
            self.backbone.eval()
        return self.backbone(x)  # type: ignore[return-value]
