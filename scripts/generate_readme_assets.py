"""Generate every visual asset embedded in README.md / docs/EXPERIMENTS.md.

For each pillar this script runs the relevant module on real demo data,
captures the visual output, and saves it under ``docs/assets/`` as a
small JPG (so the repo stays light).

Outputs (all under docs/assets/):
    Pillar 1 (tracking)        — relies on existing cctv_*.jpg thumbnails
    Pillar 2 (zero-shot)       — anomaly_clip_widget.jpg
                                  zeroshot_bus_gdino.jpg
                                  zeroshot_zidane_owlv2.jpg
                                  prompt_eval_bars.png
    Pillar 3 (3D)              — depth_bus.jpg
                                  depth_zidane.jpg
                                  stereo_compare_bus.jpg
                                  metrology_widget.jpg
    Pillar 4 (PdM + VAD)       — pdm_compare_bars.png
                                  vad_score_distribution.png
    Hero                       — hero_4panel.jpg
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ASSETS = ROOT / "docs" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)


def _save_jpg(arr: np.ndarray, path: Path, quality: int = 80) -> None:
    """Save an RGB numpy array as a quality-controlled JPG."""
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    Image.fromarray(arr).save(path, "JPEG", quality=quality, optimize=True)
    size_kb = path.stat().st_size // 1024
    print(f"    saved {path.relative_to(ROOT)} ({size_kb} KB)")


def _resize_max(arr: np.ndarray, max_side: int) -> np.ndarray:
    h, w = arr.shape[:2]
    if max(h, w) <= max_side:
        return arr
    scale = max_side / max(h, w)
    return cv2.resize(arr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Pillar 2 — zero-shot
# ---------------------------------------------------------------------------
def pillar2_anomaly_clip_widget() -> None:
    """Heatmap montage for AnomalyCLIP on 3 widget defect samples."""
    print("\n[pillar 2] AnomalyCLIP heatmap on widget defect samples")
    from mvmm.common.transforms import build_eval_transform
    from mvmm.common.viz import overlay_heatmap, side_by_side
    from mvmm.zeroshot.anomaly_clip import AnomalyCLIP

    samples = [
        ROOT / "data/sample/widget/test/defect/000.png",
        ROOT / "data/sample/widget/test/defect/001.png",
        ROOT / "data/sample/widget/test/good/000.png",
    ]
    tfm = build_eval_transform()
    # Semiconductor prompt set — chosen because the eval harness shows it
    # produces a clear gap (AUROC 1.0) on this synthetic widget data, so
    # the rendered heatmap actually reflects the discrimination signal.
    model = AnomalyCLIP(
        object_name="industrial widget",
        normal_prompts=[
            "a pristine wafer surface",
            "a die without contamination",
            "a wafer with uniform pattern",
        ],
        anomaly_prompts=[
            "a wafer with a particle defect",
            "a wafer with a scratch on the surface",
            "a die with a missing pattern",
            "a wafer with contamination",
            "a die with a bridging defect",
        ],
        windows=(2, 3),
        device="cuda",
    )

    panels: list[np.ndarray] = []
    labels = ["defect 1", "defect 2", "good"]
    for path, lab in zip(samples, labels, strict=False):
        rgb = np.asarray(Image.open(path).convert("RGB"))
        x = tfm(Image.fromarray(rgb)).unsqueeze(0)
        out = model.predict(x)
        score = float(out.image_scores[0])
        heat = overlay_heatmap(rgb, out.score_maps[0], alpha=0.55)
        cv2.putText(
            heat,
            f"{lab} | score={score:.3f}",
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
        panels.append(heat)
        print(f"    {lab:<10s} image_score={score:.4f}")

    montage = side_by_side(*panels, pad=4)
    _save_jpg(montage, ASSETS / "anomaly_clip_widget.jpg")


def pillar2_zeroshot_bus() -> None:
    """GroundingDINO overlay on bus.jpg with COCO-like prompt."""
    print("\n[pillar 2] GroundingDINO on bus.jpg")
    from mvmm.common.io import load_image
    from mvmm.tracking.detectors import GroundingDINODetector

    rgb = load_image(ROOT / "data/demo/bus.jpg")
    det = GroundingDINODetector(box_threshold=0.30, text_threshold=0.25)
    dets = det(rgb, classes=["person", "bus", "backpack", "handbag"])

    overlay = rgb.copy()
    for b, s, lab in zip(dets.boxes, dets.scores, dets.labels, strict=False):
        x1, y1, x2, y2 = (int(v) for v in b)
        name = dets.class_names[int(lab)] if 0 <= int(lab) < len(dets.class_names) else "?"
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (220, 60, 60), 3)
        cv2.putText(
            overlay,
            f"{name} {float(s):.2f}",
            (x1, max(y1 - 6, 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (220, 60, 60),
            2,
        )
    print(f"    detected {len(dets.boxes)} objects")
    _save_jpg(_resize_max(overlay, 900), ASSETS / "zeroshot_bus_gdino.jpg")


def pillar2_zeroshot_zidane() -> None:
    """OWLv2 on zidane.jpg (variety + a different backbone)."""
    print("\n[pillar 2] OWLv2 on zidane.jpg")
    from mvmm.common.io import load_image
    from mvmm.zeroshot.owl_v2 import OWLv2Detector

    rgb = load_image(ROOT / "data/demo/zidane.jpg")
    try:
        det = OWLv2Detector(score_threshold=0.15)
        dets = det(rgb, classes=["person", "soccer ball", "head", "tie"])
    except Exception as e:
        print(f"    [skip] OWLv2 not available: {type(e).__name__}: {e}")
        return

    overlay = rgb.copy()
    for b, s, lab in zip(dets.boxes, dets.scores, dets.labels, strict=False):
        x1, y1, x2, y2 = (int(v) for v in b)
        name = dets.class_names[int(lab)] if 0 <= int(lab) < len(dets.class_names) else "?"
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (60, 180, 220), 3)
        cv2.putText(
            overlay,
            f"{name} {float(s):.2f}",
            (x1, max(y1 - 6, 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (60, 180, 220),
            2,
        )
    print(f"    detected {len(dets.boxes)} objects")
    _save_jpg(_resize_max(overlay, 900), ASSETS / "zeroshot_zidane_owlv2.jpg")


def pillar2_prompt_eval_chart() -> None:
    """Bar chart of the prompt-eval AUROC numbers (from EXPERIMENTS.md)."""
    print("\n[pillar 2] prompt-eval AUROC bar chart")
    sets = ["generic", "semiconductor", "automotive", "surveillance"]
    aurocs = [0.1667, 1.0000, 1.0000, 1.0000]
    gaps = [-0.008, 0.117, 0.021, 0.095]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), dpi=140)
    colors = ["#d62728", "#2ca02c", "#2ca02c", "#2ca02c"]
    axes[0].bar(sets, aurocs, color=colors, edgecolor="black", linewidth=0.5)
    axes[0].axhline(0.5, color="gray", linestyle="--", linewidth=1, label="random baseline")
    axes[0].set_ylabel("Image AUROC")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("AnomalyCLIP — AUROC by prompt set")
    axes[0].legend(loc="lower right", fontsize=8)

    axes[1].bar(sets, gaps, color=colors, edgecolor="black", linewidth=0.5)
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_ylabel("Mean score gap (anomaly − normal)")
    axes[1].set_title("CLIP discrimination gap")

    plt.tight_layout()
    out = ASSETS / "prompt_eval_bars.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved {out.relative_to(ROOT)} ({out.stat().st_size // 1024} KB)")


# ---------------------------------------------------------------------------
# Pillar 3 — 3D
# ---------------------------------------------------------------------------
def pillar3_depth_bus() -> None:
    print("\n[pillar 3] Depth Anything v2 on bus.jpg")
    from mvmm.common.io import load_image
    from mvmm.common.viz import normalize01
    from mvmm.three_d.depth import DepthAnythingV2

    rgb = load_image(ROOT / "data/demo/bus.jpg")
    depth = DepthAnythingV2()(rgb)
    vis = (normalize01(depth) * 255).astype(np.uint8)
    color = cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    side = np.concatenate([rgb, color], axis=1)
    _save_jpg(_resize_max(side, 1400), ASSETS / "depth_bus.jpg")


def pillar3_depth_zidane() -> None:
    print("\n[pillar 3] Depth Anything v2 on zidane.jpg")
    from mvmm.common.io import load_image
    from mvmm.common.viz import normalize01
    from mvmm.three_d.depth import DepthAnythingV2

    rgb = load_image(ROOT / "data/demo/zidane.jpg")
    depth = DepthAnythingV2()(rgb)
    vis = (normalize01(depth) * 255).astype(np.uint8)
    color = cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    side = np.concatenate([rgb, color], axis=1)
    _save_jpg(_resize_max(side, 1400), ASSETS / "depth_zidane.jpg")


def pillar3_stereo_bus() -> None:
    print("\n[pillar 3] stereo-from-mono panel on bus.jpg")
    from mvmm.common.io import load_image
    from mvmm.common.viz import normalize01, side_by_side
    from mvmm.three_d.depth import DepthAnythingV2, synthesize_right_view

    rgb = load_image(ROOT / "data/demo/bus.jpg")
    depth = DepthAnythingV2()(rgb)
    right, disp = synthesize_right_view(rgb, depth, focal_px=700.0, baseline_mm=80.0)
    disp_color = cv2.cvtColor(
        cv2.applyColorMap((normalize01(disp) * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS),
        cv2.COLOR_BGR2RGB,
    )

    def _label(img: np.ndarray, text: str) -> np.ndarray:
        out = img.copy()
        cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
        cv2.putText(out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return out

    panels = [
        _label(rgb, "Left (real)"),
        _label(right, "Right (synthesized)"),
        _label(disp_color, f"Disparity ({float(disp.min()):.1f}-{float(disp.max()):.1f} px)"),
    ]
    montage = side_by_side(*panels, pad=6)
    _save_jpg(_resize_max(montage, 1600), ASSETS / "stereo_compare_bus.jpg")


def pillar3_metrology_widget() -> None:
    """Segment + dimension overlay on a widget defect image."""
    print("\n[pillar 3] metrology measurement overlay")
    from mvmm.common.io import load_image
    from mvmm.three_d.metrology.measure import dimension_from_mask
    from mvmm.three_d.metrology.segmentation import ClassicalSegmenter

    rgb = load_image(ROOT / "data/sample/widget/test/defect/000.png")
    h, w = rgb.shape[:2]
    seg = ClassicalSegmenter()
    mask = seg(rgb, points=[(w // 2, h // 2, 1)])
    try:
        dim = dimension_from_mask(mask, scale_mm_per_px=0.10)
    except ValueError as e:
        print(f"    [skip] no contour found: {e}")
        return

    overlay = rgb.copy()
    overlay[mask > 0] = (overlay[mask > 0] * 0.6 + np.array([0, 255, 0]) * 0.4).astype(np.uint8)
    if dim.rotated_box is not None:
        box = dim.rotated_box.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay, [box], isClosed=True, color=(255, 255, 0), thickness=2)
    cv2.putText(
        overlay,
        f"{dim.width_px:.1f}x{dim.height_px:.1f} px",
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    if dim.width_mm is not None:
        cv2.putText(
            overlay,
            f"{dim.width_mm:.2f}x{dim.height_mm:.2f} mm",
            (8, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
    _save_jpg(overlay, ASSETS / "metrology_widget.jpg")


# ---------------------------------------------------------------------------
# Pillar 4 — PdM + VAD charts
# ---------------------------------------------------------------------------
def pillar4_pdm_chart() -> None:
    print("\n[pillar 4] PdM 3-way comparison chart")
    models = ["TimesNet+ResNet\n(concat)", "PatchTST\n(sensor-only)", "TimesNet+ResNet\n(cross-attn)"]
    acc = [0.9667, 0.9667, 0.9333]
    auc = [0.9956, 0.9956, 1.0000]
    params = [402946, 102594, 519042]
    x = np.arange(len(models))
    width = 0.32

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), dpi=140)
    axes[0].bar(
        x - width / 2, acc, width, label="Accuracy", color="#1f77b4", edgecolor="black", linewidth=0.5
    )
    axes[0].bar(x + width / 2, auc, width, label="AUROC", color="#ff7f0e", edgecolor="black", linewidth=0.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, fontsize=9)
    axes[0].set_ylim(0.8, 1.05)
    axes[0].set_ylabel("score")
    axes[0].set_title("Multimodal PdM — accuracy vs AUROC")
    axes[0].legend(loc="lower right")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x, [p / 1000 for p in params], color="#2ca02c", edgecolor="black", linewidth=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, fontsize=9)
    axes[1].set_ylabel("trainable params (k)")
    axes[1].set_title("Model size")
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = ASSETS / "pdm_compare_bars.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved {out.relative_to(ROOT)}  ({out.stat().st_size // 1024} KB)")


def pillar4_vad_score_chart() -> None:
    """Per-frame anomaly score distribution from the VAD checkpoint."""
    print("\n[pillar 4] VAD per-frame score chart")
    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms as T

    from mvmm.vad.conv_autoencoder import ConvAutoEncoder
    from mvmm.vad.datasets import VideoFrameDataset

    ckpt = ROOT / "checkpoints/vad_ae.pt"
    if not ckpt.exists():
        print(f"    [skip] {ckpt} missing")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tfm = T.Compose([T.Resize(128), T.CenterCrop(128), T.ToTensor()])
    ds = VideoFrameDataset(
        ROOT / "data/sample/vad/test",
        transform=tfm,
        labels_csv=ROOT / "data/sample/vad/labels.csv",
    )
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    model = ConvAutoEncoder().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()

    scores, labels = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device).float()
            s = model.anomaly_score(x).cpu().numpy()
            scores.append(s)
            labels.append(np.asarray(batch["label"]))
    scores = np.concatenate(scores)
    labels = np.concatenate(labels)
    print(
        f"    {len(labels)} frames; normal mean={scores[labels == 0].mean():.5f}  "
        f"anom mean={scores[labels == 1].mean():.5f}"
    )

    fig, ax = plt.subplots(figsize=(9, 3.6), dpi=140)
    idx = np.arange(len(scores))
    ax.scatter(idx[labels == 0], scores[labels == 0], s=18, c="#1f77b4", label="normal", alpha=0.7)
    ax.scatter(idx[labels == 1], scores[labels == 1], s=18, c="#d62728", label="anomaly", alpha=0.7)
    ax.set_xlabel("frame index (sorted by video)")
    ax.set_ylabel("reconstruction MSE (= anomaly score)")
    ax.set_title("VAD frame-AE — per-frame anomaly score on synthetic CCTV test set")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = ASSETS / "vad_score_distribution.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved {out.relative_to(ROOT)}  ({out.stat().st_size // 1024} KB)")


# ---------------------------------------------------------------------------
# Hero banner — composite 2x2
# ---------------------------------------------------------------------------
def hero_banner() -> None:
    print("\n[hero] composing 2x2 banner from pillar visuals")
    panel_paths = [
        (ASSETS / "cctv_people_detection_track.jpg", "1. CCTV Tracking"),
        (ASSETS / "zeroshot_bus_gdino.jpg", "2. Zero-shot Detection"),
        (ASSETS / "depth_bus.jpg", "3. Depth Anything v2"),
        (ASSETS / "anomaly_clip_widget.jpg", "4. AnomalyCLIP heatmap"),
    ]

    target_h = 280
    cells: list[np.ndarray] = []
    for path, label in panel_paths:
        if not path.exists():
            cells.append(np.full((target_h, target_h, 3), 30, dtype=np.uint8))
            continue
        img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        scale = target_h / h
        resized = cv2.resize(img, (int(w * scale), target_h), interpolation=cv2.INTER_AREA)
        cv2.rectangle(resized, (0, 0), (resized.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(resized, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cells.append(resized)

    # Pad all cells to the same width so np.concatenate works.
    max_w = max(c.shape[1] for c in cells)
    padded = []
    for c in cells:
        if c.shape[1] < max_w:
            pad = np.full((c.shape[0], max_w - c.shape[1], 3), 30, dtype=np.uint8)
            c = np.concatenate([c, pad], axis=1)
        padded.append(c)

    top = np.concatenate([padded[0], padded[1]], axis=1)
    bot = np.concatenate([padded[2], padded[3]], axis=1)
    banner = np.concatenate([top, bot], axis=0)
    _save_jpg(banner, ASSETS / "hero_4panel.jpg", quality=82)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
TASKS = [
    pillar2_anomaly_clip_widget,
    pillar2_zeroshot_bus,
    pillar2_zeroshot_zidane,
    pillar2_prompt_eval_chart,
    pillar3_depth_bus,
    pillar3_depth_zidane,
    pillar3_stereo_bus,
    pillar3_metrology_widget,
    pillar4_pdm_chart,
    pillar4_vad_score_chart,
    hero_banner,
]


def main() -> None:
    t0 = time.time()
    for fn in TASKS:
        try:
            fn()
        except Exception as e:
            print(f"  [error] {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n[done] total {time.time() - t0:.1f}s -> {ASSETS}")


if __name__ == "__main__":  # pragma: no cover
    main()
