"""Compress activity-recognition MP4s into web-friendly clips + animated GIFs.

For each source video in ``outputs/activity/<stem>/<stem>__activity.mp4``:
    * write ``docs/videos/<stem>__activity_preview.mp4`` — downsampled,
      lower-fps mp4 (a few MB at most, plays inline on github.com via
      an HTML <video> tag in README).
    * write ``docs/assets/<stem>__activity_preview.gif`` — short
      animated preview (~10 s) that renders in any markdown viewer
      (back-compat for places that don't run <video>).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import cv2
import numpy as np
from PIL import Image


def _resize_keep_aspect(frame_bgr: np.ndarray, target_w: int) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    if w <= target_w:
        return frame_bgr
    new_h = int(h * (target_w / w))
    # Make height even (some encoders care).
    new_h -= new_h % 2
    return cv2.resize(frame_bgr, (target_w, new_h), interpolation=cv2.INTER_AREA)


def encode_mp4_preview(
    src: Path,
    dst: Path,
    target_w: int = 480,
    out_fps: float = 12.0,
    every_nth: int = 1,
) -> None:
    """Re-encode with smaller resolution + lower FPS, keeping the same codec."""
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise FileNotFoundError(src)
    in_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    sample_every = max(round(in_fps / out_fps), 1) * max(every_nth, 1)

    # Probe size from first frame.
    ok, first = cap.read()
    if not ok:
        raise RuntimeError(f"empty video: {src}")
    sample = _resize_keep_aspect(first, target_w)
    h, w = sample.shape[:2]

    dst.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(dst), fourcc, out_fps, (w, h))

    idx = 0
    written = 0
    while True:
        if idx == 0:
            frame = first
        else:
            ok, frame = cap.read()
            if not ok:
                break
        if idx % sample_every == 0:
            writer.write(_resize_keep_aspect(frame, target_w))
            written += 1
        idx += 1
    cap.release()
    writer.release()
    try:
        rel = dst.resolve().relative_to(ROOT)
    except ValueError:
        rel = dst
    print(f"  mp4  {rel}  ({w}x{h}, {written} frames @ {out_fps:.0f} FPS, {dst.stat().st_size // 1024} KB)")


def encode_gif_preview(
    src: Path,
    dst: Path,
    target_w: int = 320,
    out_fps: float = 6.0,
    max_seconds: float = 10.0,
    start_seconds: float = 0.0,
    n_colors: int = 96,
) -> None:
    """Sample ~10 s of frames, downsample, save as quantized GIF."""
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise FileNotFoundError(src)
    in_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    start_frame = int(start_seconds * in_fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    sample_every = max(round(in_fps / out_fps), 1)
    max_frames = int(max_seconds * out_fps)

    frames: list[Image.Image] = []
    idx = 0
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % sample_every == 0:
            small = _resize_keep_aspect(frame, target_w)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb).convert("P", palette=Image.ADAPTIVE, colors=n_colors)
            frames.append(pil)
        idx += 1
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames sampled from {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        dst,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / out_fps),
        loop=0,
        optimize=True,
    )
    try:
        rel = dst.resolve().relative_to(ROOT)
    except ValueError:
        rel = dst
    print(f"  gif  {rel}  ({len(frames)} frames @ {out_fps:.0f} FPS, {dst.stat().st_size // 1024} KB)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--sources",
        nargs="+",
        type=Path,
        default=[
            ROOT / "outputs/activity/people_detection/people_detection__activity.mp4",
            ROOT / "outputs/activity/store_aisle/store_aisle__activity.mp4",
        ],
    )
    p.add_argument("--video-out", type=Path, default=ROOT / "docs/videos")
    p.add_argument("--gif-out", type=Path, default=ROOT / "docs/assets")
    p.add_argument("--video-target-w", type=int, default=480)
    p.add_argument("--video-fps", type=float, default=12.0)
    p.add_argument("--gif-target-w", type=int, default=320)
    p.add_argument("--gif-fps", type=float, default=6.0)
    p.add_argument("--gif-max-seconds", type=float, default=10.0)
    p.add_argument(
        "--video-every-nth",
        type=int,
        default=1,
        help="Take every Nth source frame *before* fps downsampling — use 2 for long 60-fps clips.",
    )
    args = p.parse_args()

    for src in args.sources:
        if not src.exists():
            print(f"[skip] {src} not present")
            continue
        stem = src.stem.replace("__activity", "")
        try:
            rel_src = src.resolve().relative_to(ROOT)
        except ValueError:
            rel_src = src
        print(f"\n[encode] {rel_src}  ({src.stat().st_size // 1024} KB)")
        encode_mp4_preview(
            src,
            args.video_out / f"{stem}__activity_preview.mp4",
            target_w=args.video_target_w,
            out_fps=args.video_fps,
            every_nth=args.video_every_nth,
        )
        encode_gif_preview(
            src,
            args.gif_out / f"{stem}__activity_preview.gif",
            target_w=args.gif_target_w,
            out_fps=args.gif_fps,
            max_seconds=args.gif_max_seconds,
        )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
