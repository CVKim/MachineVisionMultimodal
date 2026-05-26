"""Lightweight IO helpers — kept dependency-minimal."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def load_image(path: str | Path, mode: str = "RGB") -> np.ndarray:
    """Read an image as a uint8 HxWxC numpy array."""
    img = Image.open(path).convert(mode)
    return np.asarray(img)


def save_image(path: str | Path, array: np.ndarray) -> None:
    """Save an HxW or HxWxC uint8 array as image."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8)).save(path)


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write a JSON file with parents created."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)
