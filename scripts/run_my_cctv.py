"""Drop-in entry point for running the full activity + hazard pipeline on
your own CCTV / factory video.

Defaults are tuned for a typical 1080p, 25–30 FPS interior camera:
    * YOLOv8s-pose for detection + ByteTrack for ids
    * rule-based activity classifier (idle / walking / working / lifting)
    * 5-second productivity buckets
    * optional restricted-zone JSON for hazard violation events
    * optional PPE prompts for GroundingDINO-based PPE compliance check
    * 8-hour shift aggregation in summary.json

This is just a thin wrapper around run_activity_recognition.py with
defaults appropriate for real industrial footage and a final
productivity-chart pass.

Run:
    python scripts/run_my_cctv.py --video /path/to/factory_floor.mp4 ^
        --zone-json configs/zones/floor_a.json ^
        --proximity-px 80 ^
        --ppe-prompts "safety helmet,high-visibility vest"
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, required=True, help="Path to your CCTV video.")
    p.add_argument("--out", type=Path, default=Path("outputs/my_cctv"))
    p.add_argument(
        "--model",
        type=str,
        default="yolov8s-pose.pt",
        help="Pose model (yolov8n-pose.pt for speed, yolov8s/m/l-pose.pt for accuracy).",
    )
    p.add_argument("--every-nth", type=int, default=1)
    p.add_argument(
        "--bucket-seconds",
        type=float,
        default=5.0,
        help="Productivity bucket width — 60 = per-minute, 3600 = hourly.",
    )
    # Hazard config — pass-through.
    p.add_argument("--zone-json", type=Path, default=None)
    p.add_argument("--proximity-px", type=float, default=0.0)
    p.add_argument("--ppe-prompts", type=str, default="")
    # DL action plug-in (optional, slow).
    p.add_argument(
        "--dl-action",
        type=str,
        default="",
        help="VideoMAE checkpoint (e.g. MCG-NJU/videomae-base-finetuned-kinetics).",
    )
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    if not args.video.exists():
        sys.exit(f"video not found: {args.video}")

    print(f"[run_my_cctv] processing {args.video.name}")
    print(f"             pose model = {args.model}, every_nth = {args.every_nth}")
    print(f"             output base = {args.out / args.video.stem}")
    if args.zone_json:
        print(f"             restricted zones from {args.zone_json}")
    if args.proximity_px > 0:
        print(f"             proximity threshold = {args.proximity_px:.0f} px")
    if args.ppe_prompts:
        print(f"             PPE check = {args.ppe_prompts}")
    if args.dl_action:
        print(f"             deep action classifier = {args.dl_action}")

    # Step 1: run main pipeline.
    cmd_main = [
        sys.executable,
        str(ROOT / "scripts" / "run_activity_recognition.py"),
        "--video",
        str(args.video),
        "--out",
        str(args.out),
        "--model",
        args.model,
        "--every-nth",
        str(args.every_nth),
        "--device",
        args.device,
    ]
    if args.zone_json:
        cmd_main += ["--zone-json", str(args.zone_json)]
    if args.proximity_px > 0:
        cmd_main += ["--proximity-px", str(args.proximity_px)]
    if args.ppe_prompts:
        cmd_main += ["--ppe-prompts", args.ppe_prompts]
    if args.dl_action:
        cmd_main += ["--dl-action", args.dl_action]
    rc = subprocess.call(cmd_main)
    if rc != 0:
        sys.exit(rc)

    # Step 2: productivity charts from the summary.
    summary_json = args.out / args.video.stem / f"{args.video.stem}__summary.json"
    if summary_json.exists():
        rc = subprocess.call(
            [
                sys.executable,
                str(ROOT / "scripts" / "plot_productivity.py"),
                "--summary",
                str(summary_json),
                "--bucket-seconds",
                str(args.bucket_seconds),
            ]
        )
        if rc != 0:
            print(f"[warn] productivity chart step exited {rc}")

    print("\n[run_my_cctv] all artifacts are under:")
    print(f"  {args.out / args.video.stem}/")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
