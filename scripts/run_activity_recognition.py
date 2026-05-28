"""Run pose-based activity recognition on a CCTV video.

Output (under ``outputs/activity/<video_stem>/``):
    <stem>__activity.mp4            annotated video (skeleton + state badge + bottom timeline)
    <stem>__states.csv              (track_id, timestamp_s, state) per frame
    <stem>__summary.json            time-in-state per track + global totals
    <stem>__timeline.png            per-track gantt of states across full video
    <stem>__time_breakdown.png      stacked bar of total seconds per state

Usage:
    python scripts/run_activity_recognition.py \\
        --video data/demo/cctv/store_aisle.mp4 \\
        --model yolov8n-pose.pt
"""

from __future__ import annotations

import argparse
import json
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

from mvmm.tracking.pose_activity import (
    STATE_COLORS_BGR,
    STATE_ORDER,
    ActivityClassifier,
    PoseTracker,
    draw_pose_track,
    draw_state_strip,
    save_state_logs_csv,
)


def _state_color_mpl(state: str) -> tuple[float, float, float]:
    b, g, r = STATE_COLORS_BGR.get(state, STATE_COLORS_BGR["unknown"])
    return (r / 255.0, g / 255.0, b / 255.0)


def plot_full_timeline(classifier: ActivityClassifier, n_frames: int, fps: float, path: Path) -> None:
    """Per-track horizontal gantt across the entire video."""
    tracks = sorted(classifier.histories.keys())
    if not tracks:
        print("  [skip] no tracks for timeline plot")
        return

    fig_h = max(1.4 + 0.32 * len(tracks), 2.4)
    fig, ax = plt.subplots(figsize=(12, fig_h), dpi=140)
    for i, tid in enumerate(tracks):
        log = classifier.histories[tid].state_log
        for j in range(1, len(log)):
            t0, s0 = log[j - 1]
            t1, _ = log[j]
            ax.barh(i, t1 - t0, left=t0, height=0.7, color=_state_color_mpl(s0), edgecolor="none")
    ax.set_yticks(range(len(tracks)))
    ax.set_yticklabels([f"#{t}" for t in tracks])
    ax.set_xlabel("video time (seconds)")
    ax.set_xlim(0, n_frames / max(fps, 1.0))
    ax.set_title(f"Per-track activity timeline  ({len(tracks)} tracks)")
    ax.grid(axis="x", alpha=0.3)
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=_state_color_mpl(s)) for s in STATE_ORDER]
    ax.legend(legend_handles, STATE_ORDER, loc="upper right", fontsize=8, ncol=len(STATE_ORDER))
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def plot_time_breakdown(classifier: ActivityClassifier, path: Path) -> None:
    """Stacked bar — per-track seconds per state + global aggregate."""
    tracks = sorted(classifier.histories.keys())
    per_track = classifier.time_in_state()
    totals = classifier.total_time_in_state()
    if not tracks:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), dpi=140, gridspec_kw={"width_ratios": [3, 1]})

    bottoms = np.zeros(len(tracks), dtype=np.float32)
    for s in STATE_ORDER:
        vals = np.array([per_track[t].get(s, 0.0) for t in tracks], dtype=np.float32)
        axes[0].bar(
            [str(t) for t in tracks],
            vals,
            bottom=bottoms,
            color=_state_color_mpl(s),
            label=s,
            edgecolor="black",
            linewidth=0.4,
        )
        bottoms += vals
    axes[0].set_ylabel("seconds")
    axes[0].set_xlabel("track id")
    axes[0].set_title("per-track time-in-state")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8)

    state_vals = [totals.get(s, 0.0) for s in STATE_ORDER]
    bar = axes[1].bar(
        STATE_ORDER,
        state_vals,
        color=[_state_color_mpl(s) for s in STATE_ORDER],
        edgecolor="black",
        linewidth=0.4,
    )
    axes[1].set_ylabel("total person-seconds")
    axes[1].set_title("aggregate")
    axes[1].grid(axis="y", alpha=0.3)
    for rect, v in zip(bar, state_vals, strict=False):
        axes[1].text(
            rect.get_x() + rect.get_width() / 2,
            v + 0.05 * max([*state_vals, 1]),
            f"{v:.1f}s",
            ha="center",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("outputs/activity"))
    p.add_argument("--model", type=str, default="yolov8n-pose.pt")
    p.add_argument("--tracker", type=str, default="bytetrack.yaml")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--conf", type=float, default=0.30)
    p.add_argument("--window", type=int, default=15)
    p.add_argument("--smooth", type=int, default=8)
    p.add_argument(
        "--every-nth",
        type=int,
        default=1,
        help="Process only every Nth frame (speed-up on long high-fps clips).",
    )
    p.add_argument(
        "--dl-action",
        type=str,
        default="",
        help="Optional VideoMAE checkpoint to plug into ActivityClassifier "
        "(e.g. MCG-NJU/videomae-base-finetuned-kinetics). Heavy — downloads ~360 MB.",
    )
    p.add_argument("--dl-clip-len", type=int, default=16)
    p.add_argument("--dl-every-n-frames", type=int, default=8)
    # Hazard detection options.
    p.add_argument(
        "--zone-json",
        type=Path,
        default=None,
        help="JSON file with a list of {name, polygon: [[x,y], ...]} restricted zones.",
    )
    p.add_argument(
        "--proximity-px",
        type=float,
        default=0.0,
        help="Pixel-distance threshold for proximity hazards. 0 disables.",
    )
    p.add_argument(
        "--ppe-prompts",
        type=str,
        default="",
        help="Comma-separated PPE classes to require (e.g. 'safety helmet,high-visibility vest'). "
        "Triggers GroundingDINO inside each person crop.",
    )
    p.add_argument("--ppe-every-n", type=int, default=30)
    args = p.parse_args()

    out_dir = args.out / args.video.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise FileNotFoundError(args.video)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    effective_fps = src_fps / max(args.every_nth, 1)
    print(f"[activity] {args.video.name}  {w}x{h}  {n_frames} frames @ {src_fps:.1f} FPS")
    print(f"           processing every {args.every_nth}th frame -> effective {effective_fps:.1f} FPS")

    # Strip height for the gantt overlay.
    strip_h = 60
    out_mp4 = out_dir / f"{args.video.stem}__activity.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(out_mp4), fourcc, max(src_fps / max(args.every_nth, 1), 1.0), (w, h + strip_h)
    )

    tracker = PoseTracker(model=args.model, device=args.device, conf=args.conf, tracker=args.tracker)

    dl_classifier = None
    if args.dl_action:
        from mvmm.tracking.action_dl import DeepActionClassifier

        print(f"[activity] loading deep action classifier: {args.dl_action}")
        dl_classifier = DeepActionClassifier(
            model_id=args.dl_action,
            device=args.device,
            clip_len=args.dl_clip_len,
            every_n_frames=args.dl_every_n_frames,
        )

    classifier = ActivityClassifier(
        fps=effective_fps,
        window_frames=args.window,
        smooth_frames=args.smooth,
        dl_classifier=dl_classifier,
    )

    hazard_detector = None
    if args.zone_json or args.proximity_px > 0 or args.ppe_prompts:
        from mvmm.tracking.analytics import PolygonZone
        from mvmm.tracking.hazard import HazardDetector

        zones: list[PolygonZone] = []
        if args.zone_json:
            zones_data = json.loads(args.zone_json.read_text(encoding="utf-8"))
            for z in zones_data:
                zones.append(PolygonZone(polygon=np.array(z["polygon"], dtype=np.float32), name=z["name"]))
            print(f"[activity] loaded {len(zones)} restricted zones from {args.zone_json}")

        ppe_detector = None
        ppe_classes = [c.strip() for c in args.ppe_prompts.split(",") if c.strip()]
        if ppe_classes:
            from mvmm.tracking.detectors import GroundingDINODetector

            print(f"[activity] PPE check: {ppe_classes}  (GroundingDINO every {args.ppe_every_n} frames)")
            ppe_detector = GroundingDINODetector(device=args.device, box_threshold=0.25)

        hazard_detector = HazardDetector(
            restricted_zones=zones,
            proximity_threshold_px=(args.proximity_px or None),
            ppe_detector=ppe_detector,
            ppe_prompts=ppe_classes,
            ppe_every_n=args.ppe_every_n,
        )

    frame_idx = 0
    written = 0
    t0 = time.time()
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if frame_idx % max(args.every_nth, 1) != 0:
            frame_idx += 1
            continue

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pose_tracks = tracker.update(rgb)
        timestamp = frame_idx / max(src_fps, 1.0)
        classifier.update(pose_tracks, timestamp, frame_rgb=rgb)

        hazard_events: list = []
        if hazard_detector is not None:
            hazard_events = hazard_detector.step(pose_tracks, rgb, timestamp)

        annot = frame_bgr.copy()
        # Render restricted zones first so skeletons draw on top.
        if hazard_detector is not None and hazard_detector.zones:
            for z in hazard_detector.zones:
                pts = z.polygon.astype(np.int32).reshape(-1, 1, 2)
                overlay = annot.copy()
                cv2.fillPoly(overlay, [pts], (0, 0, 200))
                annot = cv2.addWeighted(overlay, 0.18, annot, 0.82, 0)
                cv2.polylines(annot, [pts], isClosed=True, color=(0, 0, 200), thickness=2)
        for t in pose_tracks:
            draw_pose_track(annot, t)
        if hazard_events:
            from mvmm.tracking.hazard import draw_hazards

            draw_hazards(annot, hazard_events, pose_tracks)
        composed = draw_state_strip(
            annot, classifier, height=strip_h, seconds_window=20.0, timestamp=timestamp
        )
        writer.write(composed)
        written += 1
        if written % 50 == 0:
            print(f"   {written} frames  ({frame_idx + 1}/{n_frames})")
        frame_idx += 1

    cap.release()
    writer.release()
    elapsed = time.time() - t0
    print(
        f"[activity] wrote {written} frames in {elapsed:.1f}s ({written / max(elapsed, 1e-3):.1f} FPS effective)"
    )

    # Save CSV + JSON + charts.
    save_state_logs_csv(classifier, out_dir / f"{args.video.stem}__states.csv")
    summary = {
        "video": str(args.video),
        "resolution": [w, h],
        "src_fps": src_fps,
        "frames_total": n_frames,
        "frames_processed": written,
        "effective_fps": effective_fps,
        "wall_seconds": round(elapsed, 2),
        "tracks": sorted(classifier.histories.keys()),
        "time_in_state_s_per_track": classifier.time_in_state(),
        "total_time_in_state_s": classifier.total_time_in_state(),
    }
    if hazard_detector is not None:
        summary["hazards"] = {
            "summary": hazard_detector.summary(),
            "events": [e.as_dict() for e in hazard_detector.all_events()],
        }
    with (out_dir / f"{args.video.stem}__summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    plot_full_timeline(
        classifier, n_frames=written, fps=effective_fps, path=out_dir / f"{args.video.stem}__timeline.png"
    )
    plot_time_breakdown(classifier, path=out_dir / f"{args.video.stem}__time_breakdown.png")

    print(f"\n[activity] outputs in {out_dir}")
    print(f"  annotated video: {out_mp4}")
    print("  total time-in-state (person-seconds):")
    for s, secs in classifier.total_time_in_state().items():
        print(f"    {s:<8s} {secs:>6.1f}s")
    if hazard_detector is not None:
        h_summary = hazard_detector.summary()
        print(f"  hazards: {h_summary['n_events']} events  by_type={h_summary['by_type']}")


if __name__ == "__main__":  # pragma: no cover
    main()
