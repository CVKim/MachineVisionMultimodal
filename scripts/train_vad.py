"""Train the frame-autoencoder VAD baseline on a normals-only folder.

Run:
    python scripts/train_vad.py \
        --train-root data/sample/vad/train/normal \
        --output     checkpoints/vad_ae.pt \
        --epochs     10
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

from mvmm.vad.conv_autoencoder import ConvAutoEncoder
from mvmm.vad.datasets import VideoFrameDataset


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-root", type=Path, required=True)
    p.add_argument("--labels", type=Path, default=None)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--output", type=Path, default=Path("checkpoints/vad_ae.pt"))
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    tfm = T.Compose([T.Resize(args.image_size), T.CenterCrop(args.image_size), T.ToTensor()])
    ds = VideoFrameDataset(args.train_root, transform=tfm, labels_csv=args.labels, normals_only=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = ConvAutoEncoder(in_channels=3, base=32, latent=256).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    crit = torch.nn.MSELoss()

    print(f"[vad-train] device={device}  samples={len(ds)}  epochs={args.epochs}")
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        total = 0.0
        for batch in loader:
            x = batch["image"].to(device).float()
            recon = model(x)
            loss = crit(recon, x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * x.shape[0]
        print(f"  epoch {ep + 1}/{args.epochs}  recon_loss={total / max(len(ds), 1):.5f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f"[vad-train] saved {args.output}  ({time.time() - t0:.1f}s total)")


if __name__ == "__main__":  # pragma: no cover
    main()
