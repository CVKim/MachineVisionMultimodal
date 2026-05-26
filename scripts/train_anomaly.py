"""Standalone training entrypoint (Hydra-wrapped).

Run:
    python scripts/train_anomaly.py anomaly=patchcore data.category=bottle
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mvmm.anomaly.patchcore import PatchCore
from mvmm.common.data import MVTecADDataset
from mvmm.common.transforms import build_eval_transform


@hydra.main(version_base=None, config_path=str(ROOT / "configs"), config_name="defaults")
def main(cfg: DictConfig) -> None:
    if cfg.anomaly.name != "patchcore":
        raise NotImplementedError(f"This entrypoint only supports patchcore; got {cfg.anomaly.name}")

    cat_root = Path(cfg.paths.data) / "mvtec_ad" / cfg.data.category
    tfm = build_eval_transform(cfg.anomaly.image_size, cfg.anomaly.crop_size)
    ds = MVTecADDataset(cat_root, split="train", transform=tfm, load_masks=False)
    loader = DataLoader(
        ds, batch_size=cfg.anomaly.batch_size, num_workers=cfg.data.num_workers, shuffle=False
    )

    device = "cuda" if torch.cuda.is_available() and cfg.hardware.device == "cuda" else "cpu"
    model = PatchCore(
        layers=tuple(cfg.anomaly.layers),
        target_size=cfg.anomaly.target_size,
        coreset_ratio=cfg.anomaly.coreset_ratio,
        device=device,
        use_faiss=cfg.anomaly.use_faiss,
    )
    model.fit(loader)

    out = Path(cfg.paths.checkpoints) / f"patchcore_{cfg.data.category}.pkl"
    model.save(out)
    print(f"saved: {out}")


if __name__ == "__main__":  # pragma: no cover
    main()
