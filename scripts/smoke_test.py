"""End-to-end smoke test on the synthetic sample dataset.

Steps:
    1. Generate a tiny MVTec-shaped dataset under data/sample/widget.
    2. Train PatchCore on it (CPU-friendly, runs in seconds).
    3. Evaluate on the test split and print AUROC.

Run:
    python scripts/smoke_test.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms as T

    # 1) Generate
    from scripts.make_sample_data import main as make_sample  # type: ignore

    if not (ROOT / "data" / "sample" / "widget").exists():
        make_sample()

    from mvmm.anomaly.patchcore import PatchCore
    from mvmm.common.data import MVTecADDataset
    from mvmm.common.metrics import image_auroc
    from mvmm.common.transforms import build_eval_transform

    data_root = ROOT / "data" / "sample" / "widget"
    tfm = build_eval_transform(256, 224)

    train = MVTecADDataset(data_root, split="train", transform=tfm, load_masks=False)
    test = MVTecADDataset(
        data_root,
        split="test",
        transform=tfm,
        mask_transform=T.Compose([T.Resize(256), T.CenterCrop(224), T.PILToTensor()]),
        load_masks=True,
    )
    train_loader = DataLoader(train, batch_size=4, shuffle=False, num_workers=0)
    test_loader = DataLoader(test, batch_size=4, shuffle=False, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[smoke] device = {device}")

    t0 = time.time()
    model = PatchCore(coreset_ratio=0.25, device=device, use_faiss=False)
    model.fit(train_loader)
    t1 = time.time()
    print(f"[smoke] fit done in {t1 - t0:.2f}s; bank size = {model._bank.shape}")

    scores, labels = [], []
    for batch in test_loader:
        out = model.predict(batch["image"])
        scores.append(out.image_scores)
        labels.append(batch["label"].numpy())
    img_scores = np.concatenate(scores)
    labels = np.concatenate(labels)
    auc = image_auroc(img_scores, labels)
    print(f"[smoke] image AUROC = {auc:.4f}")
    assert auc > 0.7, f"smoke failed: AUROC too low ({auc:.3f})"
    print("[smoke] PASS")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
