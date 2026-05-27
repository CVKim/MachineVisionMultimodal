"""Generate a tiny synthetic Video Anomaly Detection dataset.

Scenario: a "factory floor camera" sees workers/forklifts moving along
clean trajectories most of the time (normal). Occasionally a red object
appears in a region it should not — that's the "anomalous" event.

Output:
    data/sample/vad/
        train/normal/<vid>/<frame>.png   (only normal clips, for AE training)
        test/normal/<vid>/<frame>.png
        test/anomaly/<vid>/<frame>.png
        labels.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


def _normal_frame(t: int, size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed * 7 + t * 13)
    img = np.full((size, size, 3), [80, 90, 110], dtype=np.float32)
    img += rng.normal(0, 5, size=img.shape)
    # A moving "worker" silhouette
    x = int(40 + (t * 6) % (size - 80))
    y = int(size // 2 + 20 * np.sin(t * 0.3))
    img[max(y - 30, 0) : y + 30, max(x - 15, 0) : x + 15] = [40, 50, 60]
    return np.clip(img, 0, 255).astype(np.uint8)


def _anomaly_frame(t: int, size: int, seed: int) -> np.ndarray:
    img = _normal_frame(t, size, seed)
    if t > 10:
        # A bright red intruder enters the bottom-right
        rx = size - 60 - int((t - 10) * 4) % 100
        ry = size - 60
        img[ry - 25 : ry + 25, rx - 25 : rx + 25] = [220, 40, 40]
    return img


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data/sample/vad"))
    p.add_argument("--frames", type=int, default=40, help="frames per clip")
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--train-clips", type=int, default=4)
    p.add_argument("--test-normal-clips", type=int, default=2)
    p.add_argument("--test-anomaly-clips", type=int, default=2)
    args = p.parse_args()

    label_rows: list[tuple[str, int, int]] = []

    def emit(folder: Path, vid: str, t: int, arr: np.ndarray, anomaly: int) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr).save(folder / f"{t:04d}.png")
        label_rows.append((vid, t, anomaly))

    for c in range(args.train_clips):
        vid = f"train_normal_{c:02d}"
        for t in range(args.frames):
            emit(args.root / "train" / "normal" / vid, vid, t, _normal_frame(t, args.size, c), 0)
    for c in range(args.test_normal_clips):
        vid = f"test_normal_{c:02d}"
        for t in range(args.frames):
            emit(args.root / "test" / "normal" / vid, vid, t, _normal_frame(t, args.size, 100 + c), 0)
    for c in range(args.test_anomaly_clips):
        vid = f"test_anomaly_{c:02d}"
        for t in range(args.frames):
            anomaly = int(t > 10)
            emit(
                args.root / "test" / "anomaly" / vid,
                vid,
                t,
                _anomaly_frame(t, args.size, 200 + c),
                anomaly,
            )

    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root / "labels.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "frame_idx", "anomaly"])
        w.writerows(label_rows)

    print(f"VAD sample data written to {args.root}  ({len(label_rows)} frames total)")


if __name__ == "__main__":  # pragma: no cover
    main()
