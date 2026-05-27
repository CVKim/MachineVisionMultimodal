"""Datasets for Video Anomaly Detection.

Generic ``VideoFrameDataset`` works with two layouts:

  A) Frame-folder (most common after frame extraction):
        root/
            <video_id>/<frame_idx>.png
        labels.csv  (optional)  with columns: video_id, frame_idx, anomaly

  B) Direct video files (will use OpenCV for streaming):
        root/<video_id>.mp4

Outputs: dict with keys "image" (torch Tensor CxHxW, normalized) and
"label" (0 / 1 if labels.csv present, else 0).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


class VideoFrameDataset:
    """Image-folder based VAD dataset.

    Args:
        root:        folder of `<video>/<frame>.png` images.
        transform:   torchvision-style transform applied to each frame.
        labels_csv:  optional CSV with columns (video_id, frame_idx, anomaly).
        normals_only: if True, drop any sample with anomaly==1 (training-time).
    """

    def __init__(
        self,
        root: str | Path,
        transform: Callable | None = None,
        labels_csv: str | Path | None = None,
        normals_only: bool = False,
    ):
        self.root = Path(root)
        self.transform = transform
        labels: dict[tuple[str, int], int] = {}
        if labels_csv is not None:
            df = pd.read_csv(labels_csv)
            for _, r in df.iterrows():
                labels[(str(r["video_id"]), int(r["frame_idx"]))] = int(r["anomaly"])
        self.labels = labels

        # Discover frames at any depth. The directory *directly* containing
        # each .png is treated as the video id, supporting:
        #   (a) root/<video>/<frame>.png                    (flat)
        #   (b) root/<class>/<video>/<frame>.png            (nested-by-class)
        samples: list[dict[str, Any]] = []
        for fp in sorted(self.root.rglob("*")):
            if fp.suffix.lower() not in IMG_EXTS:
                continue
            try:
                fidx = int(fp.stem)
            except ValueError:
                continue
            video_id = fp.parent.name
            lab = labels.get((video_id, fidx), 0)
            if normals_only and lab == 1:
                continue
            samples.append({"path": fp, "video": video_id, "frame": fidx, "label": lab})

        if not samples:
            raise RuntimeError(f"No frames discovered under {self.root}")
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        s = self.samples[idx]
        img = Image.open(s["path"]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = np.asarray(img)
        return {
            "image": img,
            "label": s["label"],
            "video": s["video"],
            "frame": s["frame"],
            "path": str(s["path"]),
        }
