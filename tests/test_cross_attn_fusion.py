"""Cross-attention multimodal fusion forward sanity."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mvmm.pdm.fusion_attn import CrossAttentionFusionModel, CrossAttnPdMConfig  # noqa: E402


def test_cross_attention_classification_forward():
    cfg = CrossAttnPdMConfig(
        n_sensor_channels=3,
        sensor_seq_len=64,
        sensor_hidden=16,
        sensor_depth=1,
        vision_backbone="resnet18",
        vision_pretrained=False,
        freeze_vision=True,
        attn_dim=32,
        attn_heads=2,
        num_classes=2,
    )
    model = CrossAttentionFusionModel(cfg).cpu()
    sensor = torch.randn(2, 64, 3)
    image = torch.randn(2, 3, 224, 224)
    logits = model(sensor, image)
    assert logits.shape == (2, 2)


def test_cross_attention_regression_forward():
    cfg = CrossAttnPdMConfig(
        n_sensor_channels=2,
        sensor_seq_len=64,
        sensor_hidden=16,
        sensor_depth=1,
        vision_backbone="resnet18",
        vision_pretrained=False,
        freeze_vision=True,
        attn_dim=32,
        attn_heads=2,
        num_classes=None,
    )
    model = CrossAttentionFusionModel(cfg).cpu()
    sensor = torch.randn(2, 64, 2)
    image = torch.randn(2, 3, 224, 224)
    out = model(sensor, image)
    assert out.shape == (2, 1)
