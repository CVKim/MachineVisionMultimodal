"""PdM model smoke test — tiny forward pass."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mvmm.pdm.fusion import MultimodalPdMModel, PdMConfig  # noqa: E402
from mvmm.pdm.timeseries import TimesNet  # noqa: E402


def test_timesnet_forward():
    net = TimesNet(input_channels=4, hidden_channels=16, depth=1, num_classes=3)
    x = torch.randn(2, 64, 4)
    out = net(x)
    assert out.shape == (2, 3)


def test_multimodal_pdm_forward():
    cfg = PdMConfig(
        n_sensor_channels=3,
        sensor_seq_len=64,
        sensor_hidden=16,
        sensor_depth=1,
        vision_backbone="resnet18",
        vision_pretrained=False,
        freeze_vision=True,
        fusion_dim=32,
        num_classes=2,
    )
    model = MultimodalPdMModel(cfg).cpu()
    sensor = torch.randn(2, 64, 3)
    image = torch.randn(2, 3, 224, 224)
    logits = model(sensor, image)
    assert logits.shape == (2, 2)
