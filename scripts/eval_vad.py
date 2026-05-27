"""Evaluate a trained VAD model — frame-level AUROC + per-clip scores.

Run:
    python scripts/eval_vad.py \
        --test-root  data/sample/vad/test \
        --labels     data/sample/vad/labels.csv \
        --checkpoint checkpoints/vad_ae.pt
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

from mvmm.common.metrics import image_auroc
from mvmm.vad.conv_autoencoder import ConvAutoEncoder
from mvmm.vad.datasets import VideoFrameDataset


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--test-root", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    tfm = T.Compose([T.Resize(args.image_size), T.CenterCrop(args.image_size), T.ToTensor()])
    ds = VideoFrameDataset(args.test_root, transform=tfm, labels_csv=args.labels, normals_only=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = ConvAutoEncoder(in_channels=3, base=32, latent=256).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()

    scores, labels, videos = [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device).float()
            s = model.anomaly_score(x).cpu().numpy()
            scores.append(s)
            labels.append(np.asarray(batch["label"]))
            videos.extend(batch["video"])
    scores = np.concatenate(scores)
    labels = np.concatenate(labels)
    auc = image_auroc(scores, labels)
    print(f"[vad-eval] frames={len(labels)}  AUROC={auc:.4f}")

    # Per-clip mean score for a quick sanity dump.
    by_video: dict[str, list[float]] = {}
    for v, s in zip(videos, scores, strict=False):
        by_video.setdefault(v, []).append(float(s))
    print("[vad-eval] per-clip mean score:")
    for v, ss in sorted(by_video.items()):
        print(f"  {v:<28s}  mean={np.mean(ss):.5f}  max={np.max(ss):.5f}")


if __name__ == "__main__":  # pragma: no cover
    main()
