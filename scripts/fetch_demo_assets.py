"""Download a set of freely-redistributable demo assets for end-to-end demos.

Assets fetched:
    * bus.jpg / zidane.jpg            — Ultralytics test images
    * bus_track.mp4                   — synthesized from bus.jpg
    * data/demo/cctv/<name>.mp4       — public CCTV-style videos (large,
                                        gitignored). With ``--videos`` you
                                        also fetch these.

Public CCTV videos are pulled (with fallbacks) from:
    * intel-iot-devkit/sample-videos      — Apache-2.0, stable for years
    * Roboflow supervision examples       — used in official supervision docs
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import cv2
import numpy as np

# Small still images — always fetched.
IMAGES = {
    "bus.jpg": "https://ultralytics.com/images/bus.jpg",
    "zidane.jpg": "https://ultralytics.com/images/zidane.jpg",
}

# Public CCTV-style videos, only fetched with --videos.
# Each entry: (output_filename, [url1, url2, ...]).
# We try URLs in order — first one that succeeds wins.
CCTV_VIDEOS: list[tuple[str, list[str]]] = [
    (
        "people_detection.mp4",
        [
            "https://github.com/intel-iot-devkit/sample-videos/raw/master/people-detection.mp4",
        ],
    ),
    (
        "store_aisle.mp4",
        [
            "https://github.com/intel-iot-devkit/sample-videos/raw/master/store-aisle-detection.mp4",
        ],
    ),
    (
        "people_walking.mp4",
        [
            "https://media.roboflow.com/supervision/video-examples/people-walking.mp4",
        ],
    ),
    (
        "vehicles.mp4",
        [
            "https://media.roboflow.com/supervision/video-examples/vehicles.mp4",
        ],
    ),
]


def _download(url: str, dst: Path) -> bool:
    if dst.exists():
        print(f"  [skip] {dst} ({dst.stat().st_size // 1024} KB) already exists")
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, dst)
        print(f"  [ok]   {url} -> {dst} ({dst.stat().st_size // 1024} KB)")
        return True
    except Exception as e:
        print(f"  [fail] {url}: {e}")
        return False


def _download_first_working(urls: list[str], dst: Path) -> bool:
    """Try each URL in turn; stop at the first successful download."""
    if dst.exists():
        print(f"  [skip] {dst} ({dst.stat().st_size // 1024} KB) already exists")
        return True
    return any(_download(url, dst) for url in urls)


def synthesize_tracking_video(
    src_image: Path,
    dst_video: Path,
    n_frames: int = 60,
    fps: int = 15,
    motion_px: int = 4,
) -> None:
    """Build a tiny synthetic video from a still image via per-frame translation.

    YOLO will detect the same objects in every frame; ByteTrack will then
    associate them across frames and assign stable track IDs.
    """
    img = cv2.imread(str(src_image))
    if img is None:
        raise RuntimeError(f"Cannot read {src_image}")
    h, w = img.shape[:2]
    pad = motion_px * (n_frames // 2 + 4)
    h_out, w_out = h - 2 * pad, w - 2 * pad
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    dst_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(dst_video), fourcc, fps, (w_out, h_out))
    try:
        for f in range(n_frames):
            dx = motion_px * (f - n_frames // 2)
            dy = int(2 * np.sin(f * 0.3))
            y0, x0 = pad + dy, pad + dx
            writer.write(img[y0 : y0 + h_out, x0 : x0 + w_out])
    finally:
        writer.release()
    print(f"  [ok]   synthesized {dst_video}  ({n_frames} frames @ {fps} FPS, {w_out}x{h_out})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/demo"))
    parser.add_argument(
        "--videos",
        action="store_true",
        help="Also fetch public CCTV demo videos (~55 MB total).",
    )
    parser.add_argument(
        "--videos-only",
        action="store_true",
        help="Skip image assets; only fetch CCTV videos.",
    )
    args = parser.parse_args()

    if not args.videos_only:
        print(f"[fetch] images -> {args.root}")
        for name, url in IMAGES.items():
            _download(url, args.root / name)

        print("[synth] building tracking video from bus.jpg")
        src = args.root / "bus.jpg"
        if src.exists():
            synthesize_tracking_video(src, args.root / "bus_track.mp4")

    if args.videos or args.videos_only:
        cctv_dir = args.root / "cctv"
        print(f"\n[fetch] CCTV videos -> {cctv_dir}")
        ok = 0
        for name, urls in CCTV_VIDEOS:
            if _download_first_working(urls, cctv_dir / name):
                ok += 1
        print(f"\n[fetch] {ok}/{len(CCTV_VIDEOS)} CCTV videos available")


if __name__ == "__main__":  # pragma: no cover
    main()
