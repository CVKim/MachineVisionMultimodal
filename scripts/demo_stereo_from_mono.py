"""Demo: monocular image + Depth Anything v2 -> synthesized right-eye view.

Outputs a side-by-side anaglyph + the per-pixel disparity map so you can
verify the stereo synthesis math.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np

from mvmm.common.io import load_image, save_image
from mvmm.common.viz import normalize01, side_by_side
from mvmm.three_d.depth import DepthAnythingV2, synthesize_right_view


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("outputs/stereo_synth"))
    p.add_argument("--focal-px", type=float, default=700.0)
    p.add_argument("--baseline-mm", type=float, default=80.0)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rgb = load_image(args.image)
    print(f"[stereo] input  : {args.image}  shape={rgb.shape}")

    print("[stereo] running Depth Anything v2 ...")
    depth = DepthAnythingV2(device=args.device)(rgb)

    print("[stereo] synthesizing right view ...")
    right, disp = synthesize_right_view(rgb, depth, focal_px=args.focal_px, baseline_mm=args.baseline_mm)

    disp_vis = (normalize01(disp) * 255).astype(np.uint8)
    disp_color = cv2.cvtColor(cv2.applyColorMap(disp_vis, cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)

    save_image(args.out / f"{args.image.stem}__left.png", rgb)
    save_image(args.out / f"{args.image.stem}__right.png", right)
    save_image(args.out / f"{args.image.stem}__disparity.png", disp_color)
    save_image(
        args.out / f"{args.image.stem}__stereo_compare.png",
        side_by_side(rgb, right, disp_color),
    )

    # Simple red/cyan anaglyph for at-a-glance 3D verification.
    anaglyph = np.zeros_like(rgb)
    anaglyph[..., 0] = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    anaglyph[..., 1] = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)
    anaglyph[..., 2] = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)
    save_image(args.out / f"{args.image.stem}__anaglyph.png", anaglyph)

    print(f"[stereo] disparity  min={float(disp.min()):.2f}  max={float(disp.max()):.2f}")
    print(
        f"[stereo] outputs    {args.out}/{args.image.stem}__{{left,right,disparity,stereo_compare,anaglyph}}.png"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
