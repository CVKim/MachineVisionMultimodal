"""Standard anomaly-detection metrics.

We follow the MVTec-AD evaluation protocol:
    - image_auroc: image-level AUROC (binary normal vs anomalous)
    - pixel_auroc: pixel-level AUROC over GT masks
    - pro_score:   Per-Region Overlap up to a FPR threshold (default 0.3)

References:
    Bergmann et al. "MVTec AD — A Comprehensive Real-World Dataset for
    Unsupervised Anomaly Detection." CVPR 2019.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import auc, roc_auc_score, roc_curve


def image_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Image-level AUROC.

    Args:
        scores: shape (N,), higher = more anomalous.
        labels: shape (N,), 0/1.
    """
    scores = np.asarray(scores).ravel()
    labels = np.asarray(labels).ravel().astype(int)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def pixel_auroc(score_maps: np.ndarray, masks: np.ndarray) -> float:
    """Pixel-level AUROC over flattened pixels.

    Args:
        score_maps: shape (N, H, W) float.
        masks:      shape (N, H, W) {0,1}.
    """
    s = np.asarray(score_maps).ravel().astype(np.float64)
    m = np.asarray(masks).ravel().astype(int)
    if m.sum() == 0 or m.sum() == m.size:
        return float("nan")
    return float(roc_auc_score(m, s))


def pro_score(score_maps: np.ndarray, masks: np.ndarray, max_fpr: float = 0.3) -> float:
    """Per-Region Overlap (PRO) score integrated up to ``max_fpr``.

    For each threshold below ``max_fpr``, average the per-connected-component
    recall across all GT regions; then take the area under that PRO-vs-FPR
    curve, normalized by ``max_fpr`` so the score lives in [0, 1].
    """
    from scipy.ndimage import label

    score_maps = np.asarray(score_maps)
    masks = np.asarray(masks).astype(int)
    n = score_maps.shape[0]

    # Pre-extract every connected component along with its image index.
    components: list[tuple[int, np.ndarray]] = []
    for i in range(n):
        lbl, num = label(masks[i])
        for cc in range(1, num + 1):
            components.append((i, np.argwhere(lbl == cc)))
    if not components:
        return float("nan")

    bg_pixels = score_maps[masks == 0]
    if bg_pixels.size == 0:
        return float("nan")

    thresholds = np.linspace(float(score_maps.min()), float(score_maps.max()), num=200)
    fprs: list[float] = []
    pros: list[float] = []
    for t in thresholds:
        binary = score_maps >= t
        fpr = float(binary[masks == 0].mean())
        if fpr > max_fpr:
            continue
        recall_sum = 0.0
        for img_idx, idx in components:
            yy, xx = idx[:, 0], idx[:, 1]
            inter = int(binary[img_idx][yy, xx].sum())
            recall_sum += inter / len(idx)
        fprs.append(fpr)
        pros.append(recall_sum / len(components))

    if not fprs:
        return float("nan")
    order = np.argsort(fprs)
    fprs_a = np.array(fprs)[order]
    pros_a = np.array(pros)[order]
    area = auc(fprs_a, pros_a)
    return float(area / max_fpr)


def best_f1_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Return (best_threshold, best_f1) for a binary score array."""
    scores = np.asarray(scores).ravel()
    labels = np.asarray(labels).ravel().astype(int)
    _fpr, tpr, thr = roc_curve(labels, scores)
    p = int(labels.sum())
    n_neg = len(labels) - p
    tp = tpr * p
    fp = _fpr * n_neg
    fn = p - tp
    f1 = np.where(tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
    best = int(np.argmax(f1))
    return float(thr[best]), float(f1[best])
