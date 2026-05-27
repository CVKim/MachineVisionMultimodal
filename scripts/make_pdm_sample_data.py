"""Generate a synthetic PdM dataset to demo the multimodal training pipeline.

Scenario: a rotating-shaft motor whose health degrades over time.

Two modalities:
    * Sensor channels (3): vibration X / Y + drawn current (1024 timesteps per
      window, 50 windows total). Healthy state has clean periodic signals;
      degraded state superimposes high-frequency noise + a 7th-harmonic
      bearing-fault tone (a textbook signature).
    * Thermal-camera frames (224x224 RGB): healthy motors run cool, degraded
      ones develop a hotspot around the bearing housing.

Outputs:
    data/sample/pdm/
        motor.csv             — per-timestep CSV (51200 rows × 5 cols + label)
        frames/000000.png ... — one frame per 1024-step window
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def _vibration(t: np.ndarray, health: float, seed: int) -> np.ndarray:
    """health in [0, 1]: 0 = healthy, 1 = failed."""
    rng = np.random.default_rng(seed)
    fund = 0.6 * np.sin(2 * np.pi * t / 64.0)  # 1x running speed
    second = 0.2 * np.sin(2 * np.pi * t / 32.0)  # 2x harmonic
    bearing = health * 0.5 * np.sin(2 * np.pi * t / 9.1)  # 7th-order bearing tone
    noise = rng.normal(0, 0.05 + 0.25 * health, size=t.shape)
    return (fund + second + bearing + noise).astype(np.float32)


def _current(t: np.ndarray, health: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 99)
    base = 1.8 + 0.05 * np.sin(2 * np.pi * t / 64.0)
    sag = health * 0.15 * np.sin(2 * np.pi * t / 11.0)  # bearing load → current sag
    return (base + sag + rng.normal(0, 0.01, size=t.shape)).astype(np.float32)


def _thermal_frame(health: float, seed: int, size: int = 224) -> np.ndarray:
    """Cool-blue motor body with a heating hotspot proportional to health."""
    rng = np.random.default_rng(seed)
    img = np.full((size, size, 3), [40, 60, 120], dtype=np.float32)
    img += rng.normal(0, 4, size=img.shape)

    yy, xx = np.mgrid[:size, :size]
    cx = size // 2 + int(rng.uniform(-10, 10))
    cy = size // 2 + int(rng.uniform(-10, 10))
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2

    hotspot_strength = 220 * health  # 0 → invisible, 1 → bright red
    hot = np.exp(-r2 / (2 * (size * 0.18) ** 2)) * hotspot_strength
    img[..., 0] += hot  # red channel
    img[..., 1] += hot * 0.5
    return np.clip(img, 0, 255).astype(np.uint8)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data/sample/pdm"))
    p.add_argument("--n-windows", type=int, default=50)
    p.add_argument("--window-len", type=int, default=1024)
    p.add_argument("--frame-size", type=int, default=224)
    args = p.parse_args()

    root = args.root
    frames_dir = root / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for w in range(args.n_windows):
        # Health monotonically grows; switch to "failed" label past the midpoint.
        health = w / max(args.n_windows - 1, 1)
        label = int(health >= 0.5)

        # Thermal frame for this window.
        frame = _thermal_frame(health, seed=w, size=args.frame_size)
        frame_rel = f"frames/{w:06d}.png"
        Image.fromarray(frame).save(root / frame_rel)

        # Per-timestep sensor rows.
        t = np.arange(args.window_len) + w * args.window_len
        vib_x = _vibration(t, health, seed=w * 2 + 0)
        vib_y = _vibration(t, health, seed=w * 2 + 1)
        cur = _current(t, health, seed=w * 3)

        for k in range(args.window_len):
            rows.append(
                {
                    "timestamp": int(t[k]),
                    "vib_x": float(vib_x[k]),
                    "vib_y": float(vib_y[k]),
                    "current": float(cur[k]),
                    "health": label,
                    "image_path": frame_rel,
                }
            )

    df = pd.DataFrame(rows)
    csv_path = root / "motor.csv"
    df.to_csv(csv_path, index=False)
    print(f"PdM dataset: {len(df)} rows, {args.n_windows} windows, frames at {frames_dir}")
    print(f"CSV: {csv_path}")
    print(df.describe())


if __name__ == "__main__":  # pragma: no cover
    main()
