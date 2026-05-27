"""Side-by-side comparison of monocular depth backends.

Runs whichever of {Depth Anything v2, Marigold, MoGe} are installed on
the same input image, writes a single horizontal panel + per-method PNG.

Run:
    python scripts/compare_depth.py --image data/demo/bus.jpg --out outputs/depth_compare
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np

from mvmm.common.io import load_image, save_image
from mvmm.common.viz import normalize01, side_by_side


def _colorize(depth: np.ndarray) -> np.ndarray:
    vis = (normalize01(depth) * 255).astype(np.uint8)
    color = cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(color, cv2.COLOR_BGR2RGB)


def _try(name: str, fn) -> tuple[str, np.ndarray | None, float | None]:
    t0 = time.time()
    try:
        d = fn()
        return name, d, time.time() - t0
    except Exception as e:
        print(f"  [skip] {name}: {type(e).__name__}: {e}")
        return name, None, None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("outputs/depth_compare"))
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    rgb = load_image(args.image)
    args.out.mkdir(parents=True, exist_ok=True)

    results: list[tuple[str, np.ndarray | None, float | None]] = []

    def _dav2():
        from mvmm.three_d.depth import DepthAnythingV2

        return DepthAnythingV2(device=args.device)(rgb)

    def _marigold():
        from mvmm.three_d.depth.monocular_extras import Marigold

        return Marigold(device=args.device)(rgb)

    def _moge():
        from mvmm.three_d.depth.monocular_extras import MoGe

        out = MoGe(device=args.device)(rgb)
        return out["depth"]

    results.append(_try("depth_anything_v2", _dav2))
    results.append(_try("marigold", _marigold))
    results.append(_try("moge", _moge))

    panels = [rgb]
    print(f"\n{'method':<22s} {'min':>8s} {'max':>8s} {'mean':>8s} {'sec':>6s}")
    for name, d, dt in results:
        if d is None:
            continue
        color = _colorize(d)
        save_image(args.out / f"{args.image.stem}__{name}.png", color)
        print(f"{name:<22s} {float(d.min()):>8.3f} {float(d.max()):>8.3f} {float(d.mean()):>8.3f} {dt:>6.2f}")
        panels.append(color)

    if len(panels) > 1:
        montage = side_by_side(*panels)
        save_image(args.out / f"{args.image.stem}__compare.png", montage)
        print(f"\n[ok] side-by-side: {args.out / f'{args.image.stem}__compare.png'}")
    else:
        print("\n[warn] no depth backend produced output — install transformers/diffusers/moge to enable.")


if __name__ == "__main__":  # pragma: no cover
    main()
