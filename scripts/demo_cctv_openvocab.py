"""GroundingDINO open-vocabulary detection on a *single* CCTV frame.

The full video would be too slow with a HF transformer detector
(GroundingDINO does not have an Ultralytics-style fast path), so we
extract one representative frame, run open-vocab detection on it with
several prompt phrasings, and dump the overlays for inspection.

This shows the *text-prompt-driven* counterpart to the YOLO closed-vocab
pipeline, on the same source footage.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np

from mvmm.tracking.detectors import GroundingDINODetector

PROMPT_SETS = {
    "coco_like": "person. car. truck. bicycle. backpack. handbag.",
    "factory_safety": "person. forklift. pallet. hard hat. safety vest.",
    "retail": "person. shopping cart. shelf. product on the shelf.",
}


def _grab_frame(video: Path, frac: float = 0.5) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(int(total * frac), 0))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame from {video}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _draw(image_rgb: np.ndarray, dets) -> np.ndarray:
    out = image_rgb.copy()
    for b, s, lab in zip(dets.boxes, dets.scores, dets.labels, strict=False):
        x1, y1, x2, y2 = (int(v) for v in b)
        name = dets.class_names[int(lab)] if 0 <= int(lab) < len(dets.class_names) else "?"
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 64, 64), 2)
        cv2.putText(
            out,
            f"{name} {float(s):.2f}",
            (x1, max(y1 - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 64, 64),
            2,
        )
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("outputs/cctv_demo/openvocab"))
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rgb = _grab_frame(args.video, frac=0.5)
    print(f"[openvocab] video={args.video.name}  mid-frame {rgb.shape}")

    det = GroundingDINODetector(device=args.device, box_threshold=0.20, text_threshold=0.15)
    print(f"\n| {'prompt-set':<16s} | {'n_dets':>6s} | top class scores (top-5)")
    print(f"|{'-' * 18}|{'-' * 8}|{'-' * 60}")

    for name, prompt_str in PROMPT_SETS.items():
        classes = [c.strip().strip(".") for c in prompt_str.split(".") if c.strip()]
        dets = det(rgb, classes=classes).filter_by_score(0.20)
        overlay = _draw(rgb, dets)
        cv2.imwrite(
            str(args.out / f"{args.video.stem}__{name}.png"),
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
        )
        # Top-5 per prompt set for the summary table.
        if len(dets.boxes):
            order = np.argsort(-dets.scores)[:5]
            tops = ", ".join(f"{dets.class_names[int(dets.labels[i])]}({dets.scores[i]:.2f})" for i in order)
        else:
            tops = "(no detections)"
        print(f"| {name:<16s} | {len(dets.boxes):>6d} | {tops}")


if __name__ == "__main__":  # pragma: no cover
    main()
