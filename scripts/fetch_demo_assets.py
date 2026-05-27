"""Download a small set of real demo assets for end-to-end inference demos.

Assets fetched (all small, freely-redistributable, single HTTPS pull):
    * bus.jpg     — Ultralytics test image with people + bus (~200 KB)
    * zidane.jpg  — Ultralytics test image with two people (~50 KB)
    * Two extra MVTec-AD style defect samples are already committed under
      data/sample/widget/.

For tracking we additionally *synthesize* a short, realistic-ish video
from the still images via small per-frame perspective warps, so YOLOv8
actually detects something on it (unlike the crude shape video).
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ASSETS = {
    "bus.jpg": "https://ultralytics.com/images/bus.jpg",
    "zidane.jpg": "https://ultralytics.com/images/zidane.jpg",
}


def _download(url: str, dst: Path) -> bool:
    if dst.exists():
        print(f"  [skip] {dst} already exists")
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, dst)
        print(f"  [ok]   {url} -> {dst} ({dst.stat().st_size // 1024} KB)")
        return True
    except Exception as e:
        print(f"  [fail] {url}: {e}")
        return False


def synthesize_tracking_video(
    src_image: Path,
    dst_video: Path,
    n_frames: int = 60,
    fps: int = 15,
    motion_px: int = 4,
) -> None:
    """Build a synthetic video from a still image by translating it frame-by-frame.

    YOLO will detect the same objects in every frame; ByteTrack will then
    associate them across frames and assign stable track IDs — that's the
    behavior we want to demo without needing a real video clip.
    """
    img = cv2.imread(str(src_image))
    if img is None:
        raise RuntimeError(f"Cannot read {src_image}")
    h, w = img.shape[:2]
    # crop the central region so the translation does not run off the edge
    pad = motion_px * (n_frames // 2 + 4)
    h_out, w_out = h - 2 * pad, w - 2 * pad
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    dst_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(dst_video), fourcc, fps, (w_out, h_out))
    try:
        for f in range(n_frames):
            # small horizontal sweep + tiny vertical wobble
            dx = motion_px * (f - n_frames // 2)
            dy = int(2 * np.sin(f * 0.3))
            y0, x0 = pad + dy, pad + dx
            frame = img[y0 : y0 + h_out, x0 : x0 + w_out]
            writer.write(frame)
    finally:
        writer.release()
    print(f"  [ok]   synthesized {dst_video}  ({n_frames} frames @ {fps} FPS, {w_out}x{h_out})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/demo"))
    args = parser.parse_args()

    print(f"[fetch] downloading demo assets to {args.root}")
    for name, url in ASSETS.items():
        _download(url, args.root / name)

    print("[synth] building a short tracking video from bus.jpg")
    src = args.root / "bus.jpg"
    if src.exists():
        synthesize_tracking_video(src, args.root / "bus_track.mp4")
    else:
        print("  [skip] bus.jpg not available — tracking demo will fall back to data/sample/track/")


if __name__ == "__main__":  # pragma: no cover
    main()
