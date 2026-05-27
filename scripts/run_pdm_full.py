"""Run every PdM model end-to-end, save predictions, plots, and metrics.

What it produces:

    outputs/pdm_results/
        summary.json                  — all metrics across the 3 models
        predictions_<model>.csv       — per-window: prob, pred, true, ts mid
        loss_curves.png               — train loss vs epoch for all 3
        sensor_signals.png            — early vs late motor signals
        thermal_frames.png            — early / mid / late thermal frames
        prediction_timeline.png       — per-window prob vs ground truth
        confusion_matrices.png        — 3 confusion matrices
        checkpoints (also in checkpoints/):
            pdm_concat.pt
            pdm_patchtst.pt
            pdm_crossattn.pt

    docs/assets/
        pdm_timeline.png              — copy of prediction_timeline.png (smaller)
        pdm_signals.png               — copy of sensor_signals.png

For each model the trainer records the per-epoch loss; we plot all three
on the same axis so the convergence speed differences are visible.

Run:
    python scripts/run_pdm_full.py --epochs 12
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset

from mvmm.common.metrics import image_auroc
from mvmm.common.transforms import build_train_transform
from mvmm.pdm.datasets import PdMSlidingWindowDataset
from mvmm.pdm.fusion import MultimodalPdMModel, PdMConfig
from mvmm.pdm.fusion_attn import CrossAttentionFusionModel, CrossAttnPdMConfig
from mvmm.pdm.patchtst import PatchTST

OUT = ROOT / "outputs" / "pdm_results"
DOCS = ROOT / "docs" / "assets"
CKPT_DIR = ROOT / "checkpoints"


@dataclass
class RunResult:
    name: str
    accuracy: float
    auroc: float
    fit_seconds: float
    n_params: int
    loss_curve: list[float] = field(default_factory=list)
    predictions: dict[str, np.ndarray] = field(default_factory=dict)
    confusion: np.ndarray | None = None
    checkpoint_path: Path | None = None


# ---------------------------------------------------------------------------
def _split_by_index(ds, ratio: float = 0.7, seed: int = 0):
    """Deterministic train/test split — keep contiguous indices so the
    timeline visualization shows a clean train/test boundary."""
    n = len(ds)
    n_tr = int(n * ratio)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    tr_idx = sorted(perm[:n_tr].tolist())
    te_idx = sorted(perm[n_tr:].tolist())
    return Subset(ds, tr_idx), Subset(ds, te_idx), tr_idx, te_idx


def _train_model(
    model,
    train_loader,
    test_loader,
    device,
    epochs,
    lr,
    use_image: bool,
    name: str,
) -> RunResult:
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    crit = torch.nn.CrossEntropyLoss()
    loss_curve: list[float] = []

    print(f"\n[{name}] train: epochs={epochs} lr={lr} use_image={use_image}")
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        total, n_seen = 0.0, 0
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
            total += float(loss) * sensor.shape[0]
            n_seen += sensor.shape[0]
        ep_loss = total / max(n_seen, 1)
        loss_curve.append(ep_loss)
        print(f"  epoch {ep + 1:2d}/{epochs}  loss={ep_loss:.5f}")
    fit_dt = time.time() - t0

    model.eval()
    probs, preds, tgts = [], [], []
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
            p = torch.softmax(logits, dim=-1)[:, 1]
            probs.append(p.cpu().numpy())
            preds.append(logits.argmax(dim=-1).cpu().numpy())
            tgts.append(tgt.cpu().numpy())
    prob = np.concatenate(probs)
    pred = np.concatenate(preds).astype(np.int64)
    tgt = np.concatenate(tgts).astype(np.int64)

    acc = float((pred == tgt).mean())
    auc = float(image_auroc(prob, tgt))
    confusion = np.zeros((2, 2), dtype=np.int64)
    for t_, p_ in zip(tgt, pred, strict=False):
        confusion[t_, p_] += 1
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  -> acc={acc:.4f}  auroc={auc:.4f}  params={n_params:,}")
    return RunResult(
        name=name,
        accuracy=acc,
        auroc=auc,
        fit_seconds=fit_dt,
        n_params=n_params,
        loss_curve=loss_curve,
        predictions={"prob": prob, "pred": pred, "true": tgt},
        confusion=confusion,
    )


# ---------------------------------------------------------------------------
def plot_loss_curves(results: list[RunResult], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4), dpi=140)
    for r in results:
        ax.plot(range(1, len(r.loss_curve) + 1), r.loss_curve, marker="o", label=r.name)
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("PdM training curves")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.relative_to(ROOT)}")


def plot_sensor_signals(table_path: Path, path: Path) -> None:
    df = pd.read_csv(table_path)
    early = df[df["health"] == 0].head(1024)
    late = df[df["health"] == 1].tail(1024)
    fig, axes = plt.subplots(3, 2, figsize=(11, 6), dpi=140, sharex=True)
    for row, channel in enumerate(["vib_x", "vib_y", "current"]):
        axes[row, 0].plot(early[channel].to_numpy(), color="#1f77b4")
        axes[row, 0].set_ylabel(channel)
        axes[row, 0].grid(alpha=0.3)
        axes[row, 1].plot(late[channel].to_numpy(), color="#d62728")
        axes[row, 1].grid(alpha=0.3)
    axes[0, 0].set_title("Early (healthy)")
    axes[0, 1].set_title("Late (failed)")
    axes[2, 0].set_xlabel("sample within window")
    axes[2, 1].set_xlabel("sample within window")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.relative_to(ROOT)}")


def plot_thermal_frames(frames_dir: Path, n_frames_total: int, path: Path) -> None:
    """Show 3 thermal frames across the health gradient."""
    idxs = [0, n_frames_total // 2, n_frames_total - 1]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.4), dpi=140)
    labels = ["frame 0\n(healthy)", f"frame {idxs[1]}\n(transition)", f"frame {idxs[2]}\n(failed)"]
    for ax, idx, lab in zip(axes, idxs, labels, strict=False):
        img_path = frames_dir / f"{idx:06d}.png"
        if not img_path.exists():
            ax.text(0.5, 0.5, "missing", ha="center", va="center")
            ax.axis("off")
            continue
        ax.imshow(np.asarray(Image.open(img_path).convert("RGB")))
        ax.set_title(lab, fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.relative_to(ROOT)}")


def plot_prediction_timeline(results: list[RunResult], te_idx: list[int], path: Path) -> None:
    """Plot per-test-window probability of "failed" for each model vs ground truth."""
    fig, ax = plt.subplots(figsize=(11, 4), dpi=140)
    # Sort by original window index so the x-axis is the data timeline.
    order = np.argsort(np.asarray(te_idx))
    sorted_idx = np.asarray(te_idx)[order]
    truth = results[0].predictions["true"][order]
    ax.fill_between(
        sorted_idx, 0, truth, step="mid", alpha=0.18, color="gray", label="ground truth (failed=1)"
    )
    for r in results:
        prob = r.predictions["prob"][order]
        ax.plot(sorted_idx, prob, marker="o", markersize=4, label=f"{r.name}  (AUROC {r.auroc:.3f})")
    ax.set_xlabel("window index (chronological)")
    ax.set_ylabel("P(failed)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("PdM — per-window predicted failure probability vs ground truth")
    ax.legend(loc="center right", fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.relative_to(ROOT)}")


def plot_confusion(results: list[RunResult], path: Path) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(3 * len(results), 3), dpi=140)
    if len(results) == 1:
        axes = [axes]
    for ax, r in zip(axes, results, strict=False):
        cm = r.confusion
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    str(int(cm[i, j])),
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                )
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["pred 0", "pred 1"])
        ax.set_yticklabels(["true 0", "true 1"])
        ax.set_title(f"{r.name}\nacc={r.accuracy:.3f}", fontsize=10)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.relative_to(ROOT)}")


def save_predictions_csv(r: RunResult, te_idx: list[int], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["window_idx", "true", "pred", "prob_failed"])
        for idx, t_, p_, prob in zip(
            te_idx, r.predictions["true"], r.predictions["pred"], r.predictions["prob"], strict=False
        ):
            w.writerow([int(idx), int(t_), int(p_), f"{float(prob):.6f}"])
    print(f"  saved {path.relative_to(ROOT)}  ({len(te_idx)} rows)")


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, default=Path("data/sample/pdm/motor.csv"))
    parser.add_argument("--frames-dir", type=Path, default=Path("data/sample/pdm/frames"))
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

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
    tr_ds, te_ds, _tr_idx, te_idx = _split_by_index(ds, ratio=0.7, seed=0)
    print(f"[setup] device={device}  full={len(ds)}  train={len(tr_ds)}  test={len(te_ds)}")
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    te_loader = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    results: list[RunResult] = []

    # 1) TimesNet + ResNet concat
    m1 = MultimodalPdMModel(PdMConfig(n_sensor_channels=3, sensor_seq_len=args.seq_len, sensor_hidden=64)).to(
        device
    )
    r1 = _train_model(m1, tr_loader, te_loader, device, args.epochs, args.lr, True, "TimesNet+ResNet concat")
    ckpt = CKPT_DIR / "pdm_concat.pt"
    torch.save(m1.state_dict(), ckpt)
    r1.checkpoint_path = ckpt
    results.append(r1)

    # 2) PatchTST sensor-only
    m2 = PatchTST(
        n_channels=3, seq_len=args.seq_len, patch_len=32, d_model=64, n_heads=4, n_layers=2, num_classes=2
    ).to(device)
    r2 = _train_model(m2, tr_loader, te_loader, device, args.epochs, args.lr, False, "PatchTST sensor-only")
    ckpt = CKPT_DIR / "pdm_patchtst.pt"
    torch.save(m2.state_dict(), ckpt)
    r2.checkpoint_path = ckpt
    results.append(r2)

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
    r3 = _train_model(
        m3, tr_loader, te_loader, device, args.epochs, args.lr, True, "TimesNet+ResNet cross-attn"
    )
    ckpt = CKPT_DIR / "pdm_crossattn.pt"
    torch.save(m3.state_dict(), ckpt)
    r3.checkpoint_path = ckpt
    results.append(r3)

    # ----------------------- charts + csv + summary -----------------------
    print("\n[plots]")
    plot_loss_curves(results, OUT / "loss_curves.png")
    plot_sensor_signals(args.table, OUT / "sensor_signals.png")
    plot_thermal_frames(args.frames_dir, n_frames_total=50, path=OUT / "thermal_frames.png")
    plot_prediction_timeline(results, te_idx, OUT / "prediction_timeline.png")
    plot_confusion(results, OUT / "confusion_matrices.png")

    print("\n[csv]")
    for r in results:
        slug = r.name.replace(" ", "_").replace("+", "").lower()
        save_predictions_csv(r, te_idx, OUT / f"predictions_{slug}.csv")

    print("\n[copy to docs/assets]")
    shutil.copy(OUT / "prediction_timeline.png", DOCS / "pdm_timeline.png")
    shutil.copy(OUT / "sensor_signals.png", DOCS / "pdm_signals.png")
    shutil.copy(OUT / "loss_curves.png", DOCS / "pdm_loss_curves.png")
    print(f"  copied to {DOCS.relative_to(ROOT)}/")

    summary = {
        "config": vars(args) | {"device": device, "n_train": len(tr_ds), "n_test": len(te_ds)},
        "models": [
            {
                "name": r.name,
                "accuracy": r.accuracy,
                "auroc": r.auroc,
                "fit_seconds": r.fit_seconds,
                "n_params": r.n_params,
                "confusion": r.confusion.tolist() if r.confusion is not None else None,
                "checkpoint": str(r.checkpoint_path) if r.checkpoint_path else None,
                "loss_curve": r.loss_curve,
            }
            for r in results
        ],
    }
    # Cast Path objects in config to strings for JSON.
    summary["config"] = {k: str(v) if isinstance(v, Path) else v for k, v in summary["config"].items()}
    with (OUT / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[summary] {OUT / 'summary.json'}")

    print(f"\n{'model':<32s} {'acc':>7s} {'AUROC':>7s} {'fit_s':>6s} {'params':>10s}")
    for r in results:
        print(f"{r.name:<32s} {r.accuracy:>7.4f} {r.auroc:>7.4f} {r.fit_seconds:>6.1f} {r.n_params:>10,d}")


if __name__ == "__main__":  # pragma: no cover
    main()
