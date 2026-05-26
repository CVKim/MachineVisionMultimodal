"""Datasets — image-folder and MVTec-AD style structures.

MVTec-AD layout assumed:
    root/<category>/train/good/*.png
    root/<category>/test/<defect_or_good>/*.png
    root/<category>/ground_truth/<defect>/*_mask.png
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _list_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS)


class ImageFolderDataset:
    """Generic image-folder dataset returning (image, path)."""

    def __init__(self, root: str | Path, transform: Callable | None = None):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Image folder not found: {self.root}")
        self.paths = _list_images(self.root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return {"image": img, "path": str(p)}


class MVTecADDataset:
    """MVTec-AD compatible dataset.

    Args:
        root: path to a single category (e.g. /data/mvtec_ad/bottle).
        split: "train" | "test".
        transform: image transform applied to RGB.
        mask_transform: transform applied to GT mask (default: resize + tensor).
        load_masks: whether to load ground-truth masks (test split only).
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform: Callable | None = None,
        mask_transform: Callable | None = None,
        load_masks: bool = True,
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.mask_transform = mask_transform
        self.load_masks = load_masks and split == "test"

        split_dir = self.root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split dir not found: {split_dir}")

        self.samples: list[dict[str, Any]] = []
        for sub in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            label_name = sub.name  # "good" or defect type
            is_anomalous = label_name != "good"
            for img_path in _list_images(sub):
                mask_path = None
                if self.load_masks and is_anomalous:
                    mp = self.root / "ground_truth" / label_name / (img_path.stem + "_mask.png")
                    if mp.exists():
                        mask_path = mp
                self.samples.append(
                    {
                        "image_path": img_path,
                        "label": int(is_anomalous),
                        "defect": label_name,
                        "mask_path": mask_path,
                    }
                )

        if not self.samples:
            raise RuntimeError(f"No samples found under {split_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        s = self.samples[idx]
        img = Image.open(s["image_path"]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        sample: dict[str, Any] = {
            "image": img,
            "label": s["label"],
            "defect": s["defect"],
            "path": str(s["image_path"]),
        }

        if self.load_masks:
            if s["mask_path"] is not None:
                mask = Image.open(s["mask_path"]).convert("L")
            else:
                # "good" test samples have no GT mask — emit an all-zero mask
                # at the *input* image resolution so the batch collates cleanly.
                mask = Image.new(
                    "L",
                    (img.shape[-1], img.shape[-2])
                    if hasattr(img, "shape")
                    else Image.open(s["image_path"]).size,
                    0,
                )
            if self.mask_transform is not None:
                mask = self.mask_transform(mask)
            sample["mask"] = mask
        return sample
