"""Compare multiple prompt sets for zero-shot anomaly detection.

Given a labeled directory (MVTec-AD-style: train/good, test/{good, defect_*})
and one or more named prompt sets in a YAML file, this script reports
image-level AUROC for each set so you can pick the operating point.

Run:
    python scripts/eval_prompts.py \
        --data-root data/mvtec_ad/bottle \
        --prompt-file configs/prompts/anomaly_clip.yaml \
        --object-name "industrial bottle"
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import yaml
from PIL import Image

from mvmm.common.data import MVTecADDataset
from mvmm.common.metrics import image_auroc
from mvmm.common.transforms import build_eval_transform
from mvmm.zeroshot.anomaly_clip import AnomalyCLIP


def _score_with_prompts(
    ds: MVTecADDataset,
    object_name: str,
    normal_prompts: list[str],
    anomaly_prompts: list[str],
    device: str,
    windows: tuple[int, ...] = (2, 3),
) -> tuple[np.ndarray, np.ndarray]:
    model = AnomalyCLIP(
        object_name=object_name,
        normal_prompts=[p.format(obj=object_name) for p in normal_prompts],
        anomaly_prompts=[p.format(obj=object_name) for p in anomaly_prompts],
        windows=windows,
        device=device,
    )
    scores: list[float] = []
    labels: list[int] = []
    tfm = build_eval_transform()
    for i in range(len(ds)):
        sample = ds[i]
        img = sample["image"]
        if not torch.is_tensor(img):
            img = tfm(Image.fromarray(np.asarray(img)))
        out = model.predict(img.unsqueeze(0))
        scores.append(float(out.image_scores[0]))
        labels.append(int(sample["label"]))
    return np.array(scores), np.array(labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--object-name", type=str, default="industrial part")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    prompts = yaml.safe_load(args.prompt_file.read_text(encoding="utf-8"))
    ds = MVTecADDataset(args.data_root, split="test", transform=build_eval_transform(), load_masks=False)
    print(f"[eval-prompts] data={args.data_root}  n={len(ds)}  object='{args.object_name}'")

    print(
        f"\n| {'prompt set':<18s} | {'normals':>7s} | {'anomalies':>9s} | {'gap':>6s} | {'AUROC':>6s} | {'sec':>5s} |"
    )
    print(f"|{'-' * 20}|{'-' * 9}|{'-' * 11}|{'-' * 8}|{'-' * 8}|{'-' * 7}|")
    for name, cfg in prompts.items():
        t0 = time.time()
        s, y = _score_with_prompts(
            ds,
            object_name=args.object_name,
            normal_prompts=cfg["normal"],
            anomaly_prompts=cfg["anomaly"],
            device=args.device,
        )
        n_mean = float(np.mean(s[y == 0])) if (y == 0).any() else float("nan")
        a_mean = float(np.mean(s[y == 1])) if (y == 1).any() else float("nan")
        auc = image_auroc(s, y)
        dt = time.time() - t0
        print(
            f"| {name:<18s} | {n_mean:>7.4f} | {a_mean:>9.4f} |"
            f" {a_mean - n_mean:>+6.3f} | {auc:>6.4f} | {dt:>5.1f} |"
        )


if __name__ == "__main__":  # pragma: no cover
    main()
