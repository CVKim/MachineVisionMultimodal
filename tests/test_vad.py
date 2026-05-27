"""Video anomaly detection — model forward + recon-error sanity."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mvmm.vad.conv_autoencoder import ConvAutoEncoder  # noqa: E402
from mvmm.vad.memae import MemAE, entropy_loss  # noqa: E402


def test_conv_autoencoder_forward_shapes():
    model = ConvAutoEncoder(in_channels=3, base=16, latent=64)
    x = torch.randn(2, 3, 128, 128)
    out = model(x)
    assert out.shape == x.shape
    assert (out >= 0).all() and (out <= 1).all()


def test_conv_autoencoder_anomaly_score_shape():
    model = ConvAutoEncoder(in_channels=3, base=16, latent=64)
    x = torch.randn(4, 3, 128, 128)
    s = model.anomaly_score(x)
    assert s.shape == (4,)


def test_memae_forward_and_entropy():
    model = MemAE(in_channels=3, base=16, latent=64, n_slots=32)
    x = torch.randn(2, 3, 128, 128)
    recon, att = model(x)
    assert recon.shape == x.shape
    e = entropy_loss(att)
    assert torch.isfinite(e)
