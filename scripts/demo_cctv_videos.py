"""Run the tracking pipeline on the public CCTV demo clips.

Reads from ``data/demo/cctv/`` (downloaded by ``scripts/fetch_demo_assets.py
--videos``) and writes annotated MP4s + per-frame JSON to
``outputs/cctv_demo/<video_stem>/``.

For each clip we also report:
    * effective FPS (frames per wall-clock second)
    * total detections, unique track IDs
    * top-3 most-frequent classes

For ``people_detection.mp4`` we additionally exercise the analytics
helpers — a polygon zone and a directional line counter — so the
zone/counting outputs land in the JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np

# What to keep per video. Empty -> all COCO classes.
VIDEO_CONFIG: dict[str, dict] = {
    "people_detection.mp4": {"classes": ["person"], "model": "yolov8s.pt", "analytics": True},
    "people_walking.mp4": {"classes": ["person"], "model": "yolov8s.pt", "analytics": False},
    "store_aisle.mp4": {"classes": ["person"], "model": "yolov8s.pt", "analytics": False},
    "vehicles.mp4": {
        "classes": ["car", "truck", "bus", "motorcycle"],
        "model": "yolov8s.pt",
        "analytics": False,
    },
}


def _frame_size(path: Path) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h


def _run_one(video: Path, out_dir: Path, cfg: dict, device: str) -> dict:
    """Run tracking on a single clip; return a results dict."""
    from mvmm.tracking import ByteTrackTracker, TrackingPipeline, build_detector
    from mvmm.tracking.analytics import LineCounter, PolygonZone

    out_dir.mkdir(parents=True, exist_ok=True)

    w, h = _frame_size(video)
    print(f"\n=== {video.name}  ({w}x{h})  model={cfg['model']}  classes={cfg['classes']}")

    det = build_detector("yolo", model=cfg["model"], device=device, conf=0.25)
    tracker = ByteTrackTracker(frame_rate=25, track_activation_threshold=0.25)

    zones: list[PolygonZone] = []
    line_counters: list[LineCounter] = []
    if cfg.get("analytics"):
        # Define a centered ground-zone polygon (lower 60% of frame).
        zones.append(
            PolygonZone(
                polygon=np.array(
                    [[0, int(0.4 * h)], [w, int(0.4 * h)], [w, h], [0, h]],
                    dtype=np.float32,
                ),
                name="lower_zone",
            )
        )
        # A directional line crossing the middle horizontally.
        line_counters.append(LineCounter(a=(0, int(0.65 * h)), b=(w, int(0.65 * h)), name="mid_line"))

    pipeline = TrackingPipeline(
        detector=det,
        tracker=tracker,
        classes=cfg["classes"] or None,
        score_threshold=0.25,
        zones=zones,
        line_counters=line_counters,
    )

    out_mp4 = out_dir / f"{video.stem}__track.mp4"
    out_json = out_dir / f"{video.stem}__track.json"

    t0 = time.time()
    stats = pipeline.process_video(video, output_video=out_mp4, output_json=out_json, show_progress=False)
    dt = time.time() - t0

    # Reload the JSON to pull class histogram quickly.
    with out_json.open(encoding="utf-8") as f:
        payload = json.load(f)
    class_counter: Counter = Counter()
    for rec in payload["per_frame"]:
        for t in rec["tracks"]:
            class_counter[t["class"]] += 1
    top3 = class_counter.most_common(3)

    result = {
        "video": video.name,
        "resolution": [w, h],
        "frames": stats.n_frames,
        "detections_total": stats.n_detections_total,
        "unique_track_ids": len(stats.track_ids_seen),
        "wall_seconds": round(dt, 2),
        "effective_fps": round(stats.n_frames / max(dt, 1e-3), 1),
        "top_classes": top3,
        "lines": payload.get("lines", []),
        "annotated_mp4": str(out_mp4),
        "per_frame_json": str(out_json),
    }
    print(
        f"  frames={result['frames']} dets={result['detections_total']} "
        f"ids={result['unique_track_ids']} fps={result['effective_fps']} "
        f"wall={result['wall_seconds']}s top={top3}"
    )
    if result["lines"]:
        print(f"  line crossings: {result['lines']}")
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data/demo/cctv"))
    p.add_argument("--out", type=Path, default=Path("outputs/cctv_demo"))
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--only", type=str, default="", help="Comma-separated subset filenames.")
    args = p.parse_args()

    if not args.root.exists():
        print(f"[err] {args.root} missing — run `python scripts/fetch_demo_assets.py --videos-only` first")
        sys.exit(1)

    targets = list(VIDEO_CONFIG.keys())
    if args.only:
        wanted = {x.strip() for x in args.only.split(",") if x.strip()}
        targets = [t for t in targets if t in wanted]

    all_results: list[dict] = []
    for name in targets:
        path = args.root / name
        if not path.exists():
            print(f"[skip] {path} not present")
            continue
        result = _run_one(path, args.out / Path(name).stem, VIDEO_CONFIG[name], args.device)
        all_results.append(result)

    args.out.mkdir(parents=True, exist_ok=True)
    summary = args.out / "summary.json"
    with summary.open("w", encoding="utf-8") as f:
        json.dump({"results": all_results}, f, indent=2, ensure_ascii=False)
    print(f"\n[done] summary -> {summary}")


if __name__ == "__main__":  # pragma: no cover
    main()
