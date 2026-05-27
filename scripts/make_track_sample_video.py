"""Generate a tiny 'CCTV-style' sample video so tracking demo runs offline.

Creates a 5-second 320x240 mp4 with a couple of synthetic 'people' shapes
moving across the frame. Useful for smoke-testing the tracking pipeline
without needing real footage.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("data/sample/track/sample.mp4"))
    p.add_argument("--seconds", type=int, default=5)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--w", type=int, default=320)
    p.add_argument("--h", type=int, default=240)
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.out), fourcc, args.fps, (args.w, args.h))
    n = args.seconds * args.fps

    # Two "people" silhouettes: a dark-clothed person + a high-vis worker.
    for f in range(n):
        bg = np.full((args.h, args.w, 3), [120, 130, 140], dtype=np.uint8)
        # add a slight floor pattern
        bg[args.h // 2 :, :] = [110, 120, 130]

        # Person A: dark, moves left → right
        ax = int(20 + (f / n) * (args.w - 60))
        ay = args.h // 2
        cv2.rectangle(bg, (ax - 8, ay - 25), (ax + 8, ay + 25), (40, 40, 50), -1)  # body
        cv2.circle(bg, (ax, ay - 35), 8, (50, 50, 60), -1)  # head

        # Person B: hi-vis vest, moves right → left
        bx = int(args.w - 20 - (f / n) * (args.w - 60))
        by = args.h // 2 + 20
        cv2.rectangle(bg, (bx - 8, by - 25), (bx + 8, by + 25), (40, 200, 240), -1)
        cv2.circle(bg, (bx, by - 35), 8, (50, 60, 70), -1)

        writer.write(bg)
    writer.release()
    print(f"Sample CCTV video written to {args.out}  ({n} frames, {args.fps} FPS)")


if __name__ == "__main__":  # pragma: no cover
    main()
