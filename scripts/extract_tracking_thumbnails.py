"""Extract a few representative frames from each tracked output video.

Lets you eyeball the tracker without playing the full mp4 — useful for
README screenshots / PR review / commit artifacts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def extract_frames(video_path: Path, out_dir: Path, n: int = 3) -> list[Path]:
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        return []
    indices = [int(total * frac) for frac in [0.1, 0.5, 0.9]][:n]
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for _k, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        dst = out_dir / f"{video_path.stem}__frame{idx:05d}.jpg"
        cv2.imwrite(str(dst), frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        written.append(dst)
        print(f"  [ok] {dst}  ({frame.shape[1]}x{frame.shape[0]})")
    cap.release()
    return written


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("outputs/cctv_demo"))
    p.add_argument("--out", type=Path, default=Path("outputs/cctv_demo/thumbnails"))
    p.add_argument("--n", type=int, default=3)
    args = p.parse_args()

    videos = sorted(args.root.rglob("*__track.mp4"))
    if not videos:
        print(f"[err] no *__track.mp4 found under {args.root}")
        return

    print(f"[thumbnails] {len(videos)} videos -> {args.out}")
    for v in videos:
        extract_frames(v, args.out, n=args.n)


if __name__ == "__main__":  # pragma: no cover
    main()
