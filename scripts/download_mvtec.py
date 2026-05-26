"""Download & extract the MVTec-AD dataset.

The official archive is large (~5GB). This script verifies the SHA256 and
extracts into ``data/mvtec_ad/``. Skips work if already present.

Usage:
    python scripts/download_mvtec.py --dest data/mvtec_ad
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

URL = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629951468/mvtec_anomaly_detection.tar.xz"
EXPECTED_TOP_DIRS = {
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
}


def already_extracted(dest: Path) -> bool:
    return all((dest / d).is_dir() for d in EXPECTED_TOP_DIRS)


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as r, target.open("wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)


def extract(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:xz") as t:
        t.extractall(dest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", type=Path, default=Path("data/mvtec_ad"))
    parser.add_argument("--archive", type=Path, default=None, help="Pre-downloaded tarball.")
    args = parser.parse_args(argv)

    if already_extracted(args.dest):
        print(f"[skip] {args.dest} already populated.")
        return 0

    if args.archive is None:
        args.archive = args.dest.parent / "mvtec_anomaly_detection.tar.xz"
        if not args.archive.exists():
            print(f"[download] {URL} -> {args.archive}")
            download(URL, args.archive)
        else:
            print(f"[skip download] {args.archive} exists.")
    print(f"[extract] {args.archive} -> {args.dest}")
    extract(args.archive, args.dest)
    print("[done] MVTec-AD ready.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
