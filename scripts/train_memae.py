"""Train the Memory-Augmented AutoEncoder for VAD with entropy regularization.

Run:
    python scripts/train_memae.py \
        --train-root data/sample/vad/train/normal \
        --output checkpoints/vad_memae.pt --epochs 15
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

from mvmm.vad.datasets import VideoFrameDataset
from mvmm.vad.memae import MemAE, entropy_loss


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-root", type=Path, required=True)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--n-slots", type=int, default=2000)
    p.add_argument("--entropy-weight", type=float, default=2e-4)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--output", type=Path, default=Path("checkpoints/vad_memae.pt"))
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    tfm = T.Compose([T.Resize(args.image_size), T.CenterCrop(args.image_size), T.ToTensor()])
    ds = VideoFrameDataset(args.train_root, transform=tfm, normals_only=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = MemAE(in_channels=3, n_slots=args.n_slots).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    mse = torch.nn.MSELoss()

    print(f"[memae-train] samples={len(ds)} device={device} n_slots={args.n_slots}")
    t0 = time.time()
    for ep in range(args.epochs):
        total_r, total_e, n_seen = 0.0, 0.0, 0
        for batch in loader:
            x = batch["image"].to(device).float()
            recon, att = model(x)
            recon_loss = mse(recon, x)
            ent = entropy_loss(att)
            loss = recon_loss + args.entropy_weight * ent
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_r += float(recon_loss) * x.shape[0]
            total_e += float(ent) * x.shape[0]
            n_seen += x.shape[0]
        print(
            f"  epoch {ep + 1}/{args.epochs}  recon={total_r / max(n_seen, 1):.5f}  "
            f"entropy={total_e / max(n_seen, 1):.4f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f"[memae-train] saved {args.output}  ({time.time() - t0:.1f}s total)")


if __name__ == "__main__":  # pragma: no cover
    main()
