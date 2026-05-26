"""Generic PdM dataset.

CSV / Parquet layout (one row per timestep, multivariate sensor):
    timestamp, sensor1, sensor2, ..., sensorK, label_or_rul, image_path

Sliding windows of length ``seq_len`` are emitted as samples; the image
corresponding to the window's last timestep is loaded if provided.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image


class PdMSlidingWindowDataset:
    """Sliding window dataset that emits (sensor_window, image, target).

    Args:
        table_path:   .csv / .parquet path.
        seq_len:      window length in timesteps.
        stride:       shift between consecutive windows.
        sensor_cols:  list of column names for sensor channels.
        target_col:   column name for label / RUL.
        image_col:    column name holding a file path (relative to table dir).
        image_root:   override base path for image_col.
        transform:    image transform.
    """

    def __init__(
        self,
        table_path: str | Path,
        seq_len: int,
        sensor_cols: list[str],
        target_col: str,
        image_col: str | None = "image_path",
        image_root: str | Path | None = None,
        stride: int = 1,
        transform: Callable | None = None,
    ):
        self.table_path = Path(table_path)
        if self.table_path.suffix == ".parquet":
            self.df = pd.read_parquet(self.table_path)
        else:
            self.df = pd.read_csv(self.table_path)
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        self.sensor_cols = sensor_cols
        self.target_col = target_col
        self.image_col = image_col
        self.image_root = Path(image_root) if image_root else self.table_path.parent
        self.transform = transform

        if len(self.df) < self.seq_len:
            raise ValueError(f"Table has only {len(self.df)} rows but seq_len={self.seq_len}")
        self._starts = list(range(0, len(self.df) - self.seq_len + 1, self.stride))

    def __len__(self) -> int:
        return len(self._starts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        s = self._starts[idx]
        e = s + self.seq_len
        window = self.df.iloc[s:e]
        sensor = window[self.sensor_cols].to_numpy(dtype=np.float32)
        target = float(window[self.target_col].iloc[-1])

        image = None
        if self.image_col and self.image_col in window.columns:
            rel = window[self.image_col].iloc[-1]
            if isinstance(rel, str) and rel:
                p = self.image_root / rel
                image = Image.open(p).convert("RGB")
                if self.transform is not None:
                    image = self.transform(image)

        return {"sensor": sensor, "image": image, "target": target}
