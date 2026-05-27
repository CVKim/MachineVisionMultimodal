"""One-shot inference dump across all modules.

Runs every module that has a pretrained or training-free entry point
on the inputs in a directory and writes a flat ``outputs/dump/<ts>/``
tree containing:

    images/
        <name>__zeroshot.png   — open-vocab detection overlay
        <name>__depth.png      — colorized depth
        <name>__depth.ply      — point cloud
        <name>__patchcore.png  — heatmap (if PatchCore checkpoint provided)
        <name>__clip.png       — AnomalyCLIP zero-shot AD heatmap
    videos/
        <name>__track.mp4      — tracking overlay
        <name>__track.json     — per-frame track records
        <name>__vad.csv        — frame anomaly scores
    summary.json               — what ran, where, and what failed

Run:
    python scripts/dump_all_results.py --inputs data/sample --out outputs/dump
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
VID_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


# --------------------------------------------------------------------------- helpers
def _safe(name: str, fn, *args, summary: dict, **kwargs):
    """Run a single module; capture failure without aborting the whole dump."""
    t0 = time.time()
    try:
        out = fn(*args, **kwargs)
        summary[name] = {"ok": True, "dt": round(time.time() - t0, 2), "detail": out}
        print(f"  [ok]   {name:<22s} {time.time() - t0:5.1f}s")
        return out
    except Exception as e:
        summary[name] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        print(f"  [skip] {name:<22s} -- {type(e).__name__}: {e}")
        return None


def _list_inputs(root: Path) -> tuple[list[Path], list[Path]]:
    imgs, vids = [], []
    for p in root.rglob("*"):
        if p.suffix.lower() in IMG_EXTS:
            imgs.append(p)
        elif p.suffix.lower() in VID_EXTS:
            vids.append(p)
    return sorted(imgs), sorted(vids)


# --------------------------------------------------------------------------- per-image
def dump_per_image(img_path: Path, out_dir: Path, classes: list[str], device: str, summary: dict) -> None:
    import cv2

    from mvmm.common.io import load_image
    from mvmm.common.viz import normalize01

    name = img_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb = load_image(img_path)

    # --- depth (Depth Anything v2)
    def _depth():
        from mvmm.three_d.depth import DepthAnythingV2

        d = DepthAnythingV2(device=device)(rgb)
        vis = (normalize01(d) * 255).astype(np.uint8)
        col = cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)
        cv2.imwrite(str(out_dir / f"{name}__depth.png"), col)
        return {"min": float(d.min()), "max": float(d.max()), "shape": list(d.shape)}

    _safe(f"depth:{name}", _depth, summary=summary)

    # --- zero-shot detection (GroundingDINO)
    def _zeroshot():
        from mvmm.tracking.detectors import GroundingDINODetector

        det = GroundingDINODetector(device=device)
        d = det(rgb, classes=classes)
        overlay = rgb.copy()
        for b, s, lab in zip(d.boxes, d.scores, d.labels, strict=False):
            x1, y1, x2, y2 = (int(v) for v in b)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 64, 64), 2)
            cls = d.class_names[int(lab)] if 0 <= int(lab) < len(d.class_names) else "?"
            cv2.putText(
                overlay,
                f"{cls} {float(s):.2f}",
                (x1, max(y1 - 4, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 64, 64),
                1,
            )
        cv2.imwrite(str(out_dir / f"{name}__zeroshot.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        return {"n_detections": len(d.boxes)}

    _safe(f"zeroshot:{name}", _zeroshot, summary=summary)

    # --- AnomalyCLIP zero-shot AD heatmap
    def _clip_ad():
        from PIL import Image

        from mvmm.common.transforms import build_eval_transform
        from mvmm.common.viz import overlay_heatmap
        from mvmm.zeroshot.anomaly_clip import AnomalyCLIP

        tfm = build_eval_transform()
        x = tfm(Image.fromarray(rgb)).unsqueeze(0)
        clip = AnomalyCLIP(object_name=classes[0] if classes else "industrial part", device=device)
        out = clip.predict(x)
        heat = overlay_heatmap(rgb, out.score_maps[0])
        cv2.imwrite(str(out_dir / f"{name}__clip.png"), cv2.cvtColor(heat, cv2.COLOR_RGB2BGR))
        return {"image_score": float(out.image_scores[0])}

    _safe(f"anomalyclip:{name}", _clip_ad, summary=summary)


# --------------------------------------------------------------------------- per-video
def dump_per_video(vid_path: Path, out_dir: Path, classes: list[str], device: str, summary: dict) -> None:
    name = vid_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    def _track():
        from mvmm.tracking import ByteTrackTracker, TrackingPipeline, build_detector

        det = build_detector("yolo", model="yolov8n.pt", device=device)
        tracker = ByteTrackTracker()
        pipeline = TrackingPipeline(detector=det, tracker=tracker, classes=classes or None)
        stats = pipeline.process_video(
            vid_path,
            output_video=out_dir / f"{name}__track.mp4",
            output_json=out_dir / f"{name}__track.json",
            show_progress=False,
        )
        return stats.as_dict()

    _safe(f"track:{name}", _track, summary=summary)


# --------------------------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", type=Path, default=Path("data/sample"))
    p.add_argument("--out", type=Path, default=Path("outputs/dump"))
    p.add_argument("--classes", type=str, default="person,forklift,pallet,box,defect,scratch")
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_root = args.out / ts
    img_out = out_root / "images"
    vid_out = out_root / "videos"

    imgs, vids = _list_inputs(args.inputs)
    print(f"[dump] inputs={args.inputs}  images={len(imgs)}  videos={len(vids)}")
    print(f"[dump] writing to {out_root}")

    summary: dict = {"images": {}, "videos": {}}
    for img in imgs:
        print(f"\n[image] {img.relative_to(args.inputs)}")
        dump_per_image(img, img_out, classes, args.device, summary["images"])
    for vid in vids:
        print(f"\n[video] {vid.relative_to(args.inputs)}")
        dump_per_video(vid, vid_out, classes, args.device, summary["videos"])

    out_root.mkdir(parents=True, exist_ok=True)
    with (out_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[dump] done  ->  {out_root / 'summary.json'}")


if __name__ == "__main__":  # pragma: no cover
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
