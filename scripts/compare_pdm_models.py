"""Compare three PdM model families on the synthetic motor data.

Models:
    1. TimesNet + ResNet late-fusion (mvmm.pdm.fusion.MultimodalPdMModel)
    2. PatchTST (sensor-only)        (mvmm.pdm.patchtst.PatchTST)
    3. TimesNet + ResNet cross-attn  (mvmm.pdm.fusion_attn.CrossAttentionFusionModel)

Each is trained for the same epoch budget on the same data; we report
accuracy + AUROC on the same held-out split.

Run:
    python scripts/compare_pdm_models.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from mvmm.common.metrics import image_auroc
from mvmm.common.transforms import build_train_transform
from mvmm.pdm.datasets import PdMSlidingWindowDataset
from mvmm.pdm.fusion import MultimodalPdMModel, PdMConfig
from mvmm.pdm.fusion_attn import CrossAttentionFusionModel, CrossAttnPdMConfig
from mvmm.pdm.patchtst import PatchTST


def _split(ds, ratio=0.7, seed=0):
    n_tr = int(len(ds) * ratio)
    n_te = len(ds) - n_tr
    return random_split(ds, [n_tr, n_te], generator=torch.Generator().manual_seed(seed))


def _train_eval(model, train_loader, test_loader, device, epochs, lr, use_image: bool):
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    crit = torch.nn.CrossEntropyLoss()
    t0 = time.time()

    for _ep in range(epochs):
        model.train()
        for batch in train_loader:
            sensor = batch["sensor"].to(device).float()
            tgt = batch["target"].long().to(device)
            if use_image:
                img = batch["image"]
                if img is None or (hasattr(img, "numel") and img.numel() == 0):
                    img = torch.zeros(sensor.shape[0], 3, 224, 224, device=device)
                else:
                    img = img.to(device).float()
                logits = model(sensor, img)
            else:
                logits = model(sensor)
            loss = crit(logits, tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
    fit_dt = time.time() - t0

    model.eval()
    preds, probs, tgts = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            sensor = batch["sensor"].to(device).float()
            tgt = batch["target"].long().to(device)
            if use_image:
                img = batch["image"]
                if img is None or (hasattr(img, "numel") and img.numel() == 0):
                    img = torch.zeros(sensor.shape[0], 3, 224, 224, device=device)
                else:
                    img = img.to(device).float()
                logits = model(sensor, img)
            else:
                logits = model(sensor)
            prob = torch.softmax(logits, dim=-1)[:, 1]
            preds.append(logits.argmax(dim=-1).cpu().numpy())
            probs.append(prob.cpu().numpy())
            tgts.append(tgt.cpu().numpy())

    pred = np.concatenate(preds)
    prob = np.concatenate(probs)
    tgt_arr = np.concatenate(tgts)
    return {
        "accuracy": float((pred == tgt_arr).mean()),
        "auroc": float(image_auroc(prob, tgt_arr)),
        "fit_sec": fit_dt,
        "n_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--table", type=Path, default=Path("data/sample/pdm/motor.csv"))
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--stride", type=int, default=512)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    sensor_cols = ["vib_x", "vib_y", "current"]
    ds = PdMSlidingWindowDataset(
        table_path=args.table,
        seq_len=args.seq_len,
        sensor_cols=sensor_cols,
        target_col="health",
        stride=args.stride,
        transform=build_train_transform(),
    )
    tr_ds, te_ds = _split(ds, ratio=0.7)
    print(f"[compare-pdm] train={len(tr_ds)} test={len(te_ds)}")
    tr = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    te = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    rows: list[tuple[str, dict]] = []

    # 1) TimesNet + ResNet late-fusion concat
    m1 = MultimodalPdMModel(PdMConfig(n_sensor_channels=3, sensor_seq_len=args.seq_len, sensor_hidden=64)).to(
        device
    )
    rows.append(
        ("TimesNet+ResNet (concat)", _train_eval(m1, tr, te, device, args.epochs, args.lr, use_image=True))
    )

    # 2) PatchTST (sensor-only)
    m2 = PatchTST(
        n_channels=3, seq_len=args.seq_len, patch_len=32, d_model=64, n_heads=4, n_layers=2, num_classes=2
    ).to(device)
    rows.append(
        ("PatchTST (sensor-only)", _train_eval(m2, tr, te, device, args.epochs, args.lr, use_image=False))
    )

    # 3) Cross-attention fusion
    m3 = CrossAttentionFusionModel(
        CrossAttnPdMConfig(
            n_sensor_channels=3,
            sensor_seq_len=args.seq_len,
            sensor_hidden=64,
            attn_dim=128,
            attn_heads=4,
            vision_pretrained=True,
            freeze_vision=True,
        )
    ).to(device)
    rows.append(
        (
            "TimesNet+ResNet (cross-attn)",
            _train_eval(m3, tr, te, device, args.epochs, args.lr, use_image=True),
        )
    )

    print(f"\n| {'model':<32s} | {'accuracy':>9s} | {'AUROC':>7s} | {'fit_s':>6s} | {'params':>10s} |")
    print(f"|{'-' * 34}|{'-' * 11}|{'-' * 9}|{'-' * 8}|{'-' * 12}|")
    for name, m in rows:
        print(
            f"| {name:<32s} | {m['accuracy']:>9.4f} | {m['auroc']:>7.4f} | "
            f"{m['fit_sec']:>6.1f} | {m['n_params']:>10,d} |"
        )


if __name__ == "__main__":  # pragma: no cover
    main()
