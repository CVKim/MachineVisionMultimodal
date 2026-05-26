"""Evaluate a saved PatchCore checkpoint on the test split.

Run:
    python scripts/eval_anomaly.py --checkpoint checkpoints/patchcore_bottle.pkl \
        --data-root data/mvtec_ad/bottle
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

from mvmm.anomaly.patchcore import PatchCore
from mvmm.common.data import MVTecADDataset
from mvmm.common.metrics import image_auroc, pixel_auroc, pro_score
from mvmm.common.transforms import build_eval_transform


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    tfm = build_eval_transform(args.image_size, args.crop_size)
    mask_tfm = T.Compose([T.Resize(args.image_size), T.CenterCrop(args.crop_size), T.PILToTensor()])
    ds = MVTecADDataset(args.data_root, split="test", transform=tfm, mask_transform=mask_tfm)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=4, shuffle=False)

    model = PatchCore(device="cuda" if torch.cuda.is_available() else "cpu")
    model.load(args.checkpoint)

    img_scores, labels, pix_scores, pix_masks = [], [], [], []
    for batch in loader:
        out = model.predict(batch["image"])
        img_scores.append(out.image_scores)
        labels.append(batch["label"].numpy())
        if "mask" in batch:
            pix_scores.append(out.score_maps)
            pix_masks.append((batch["mask"].squeeze(1).numpy() > 0).astype(np.uint8))

    s = np.concatenate(img_scores)
    y = np.concatenate(labels)
    print(f"image AUROC: {image_auroc(s, y):.4f}")
    if pix_scores:
        sm = np.concatenate(pix_scores)
        gm = np.concatenate(pix_masks)
        print(f"pixel AUROC: {pixel_auroc(sm, gm):.4f}")
        print(f"PRO score:   {pro_score(sm, gm):.4f}")


if __name__ == "__main__":  # pragma: no cover
    main()
