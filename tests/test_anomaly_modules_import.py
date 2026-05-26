"""Lightweight import + interface tests — no GPU/network required."""

from __future__ import annotations

import numpy as np

from mvmm.anomaly.base import AnomalyDetector, AnomalyResult
from mvmm.anomaly.hybrid import RuleBasedScorer, fuse_rule_and_dl


def test_anomaly_result_dataclass():
    r = AnomalyResult(image_scores=np.array([0.1, 0.2]), score_maps=np.zeros((2, 8, 8)))
    d = r.as_dict()
    assert "image_scores" in d and "score_maps" in d


def test_rulebased_scorer_returns_hxw():
    img = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    s = RuleBasedScorer().score(img)
    assert s.shape == (64, 64)
    assert s.dtype == np.float32


def test_fuse_rule_and_dl_shapes():
    img_h, img_w = 32, 32
    rm = np.random.rand(img_h, img_w).astype(np.float32)
    dl = AnomalyResult(
        image_scores=np.array([0.0]),
        score_maps=np.random.rand(1, img_h, img_w).astype(np.float32),
    )
    fused = fuse_rule_and_dl(rm, dl, alpha=0.3)
    assert fused.score_maps.shape == (1, img_h, img_w)
    assert fused.image_scores.shape == (1,)


def test_anomaly_detector_is_abstract():
    assert AnomalyDetector.__abstractmethods__
