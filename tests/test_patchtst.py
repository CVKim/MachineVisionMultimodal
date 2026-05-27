"""PatchTST forward + shape sanity."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mvmm.pdm.patchtst import PatchTST  # noqa: E402


def test_patchtst_classification_forward_shape():
    model = PatchTST(
        n_channels=3, seq_len=128, patch_len=16, d_model=32, n_heads=2, n_layers=1, num_classes=2
    )
    x = torch.randn(4, 128, 3)
    out = model(x)
    assert out.shape == (4, 2)


def test_patchtst_regression_outputs_scalar():
    model = PatchTST(n_channels=2, seq_len=64, patch_len=16, num_classes=None)
    x = torch.randn(2, 64, 2)
    assert model(x).shape == (2, 1)


def test_patchtst_seq_must_divide_patch():
    with pytest.raises(ValueError):
        PatchTST(n_channels=1, seq_len=100, patch_len=7)


def test_patchtst_encode_features():
    model = PatchTST(n_channels=2, seq_len=64, patch_len=16, d_model=16, n_heads=2, n_layers=1)
    feats = model.encode_features(torch.randn(3, 64, 2))
    assert feats.shape == (3, 2 * 16)  # n_channels * d_model
