"""Zero-shot AnomalyCLIP demo: compare prompt sets across domains.

Run:
    python scripts/demo_zero_shot_prompts.py

What it shows:
    For one defective and one clean image, we score with three prompt
    flavors — generic, semiconductor-specific, automotive-specific —
    to demonstrate how prompt phrasing shifts the anomaly score and
    therefore the operating point you'd ship to a line.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from PIL import Image

from mvmm.common.transforms import build_eval_transform
from mvmm.zeroshot.anomaly_clip import AnomalyCLIP

PROMPT_SETS = {
    "generic": (
        ["a flawless {obj}", "a clean {obj}"],
        ["a defective {obj}", "a {obj} with a scratch", "a damaged {obj}"],
    ),
    "semiconductor": (
        [
            "a pristine wafer surface",
            "a die without contamination",
            "a wafer with uniform pattern",
        ],
        [
            "a wafer with a particle defect",
            "a wafer with a scratch on the surface",
            "a die with a missing pattern",
            "a wafer with contamination",
            "a die with a bridging defect",
        ],
    ),
    "automotive": (
        [
            "a metal automotive part with smooth surface",
            "a clean machined part",
            "a flawless cast aluminum component",
        ],
        [
            "an automotive part with a scratch",
            "a cast part with a porosity defect",
            "a machined part with a burr",
            "a metal part with a dent",
            "a part with corrosion",
        ],
    ),
}


def run_demo(
    images: dict[str, Path],
    object_name: str = "industrial part",
    device: str = "cuda",
) -> None:
    tfm = build_eval_transform()
    print(f"{'prompt-set':<16} {'image':<22} {'expected':<10} {'image-score':>12}")
    print("-" * 64)

    for prompt_label, (normal, anomaly) in PROMPT_SETS.items():
        model = AnomalyCLIP(
            object_name=object_name,
            normal_prompts=[p.replace("{obj}", object_name) for p in normal],
            anomaly_prompts=[p.replace("{obj}", object_name) for p in anomaly],
            windows=(2, 3),  # fewer windows for demo speed
            device=device,
        )
        for tag, path in images.items():
            rgb = np.asarray(Image.open(path).convert("RGB"))
            x = tfm(Image.fromarray(rgb)).unsqueeze(0)
            out = model.predict(x)
            print(
                f"{prompt_label:<16} {tag:<22} "
                f"{'anomaly' if 'defect' in tag else 'normal ':<10} "
                f"{float(out.image_scores[0]):>12.4f}"
            )
        print()


def main() -> None:
    sample_dir = ROOT / "data" / "sample" / "widget"
    if not sample_dir.exists():
        from scripts.make_sample_data import main as make_sample  # type: ignore

        make_sample()

    images = {
        "good/000": sample_dir / "test" / "good" / "000.png",
        "defect/000": sample_dir / "test" / "defect" / "000.png",
    }
    run_demo(images)


if __name__ == "__main__":  # pragma: no cover
    main()
