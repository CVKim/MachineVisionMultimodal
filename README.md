# MachineVisionMultimodal (`mvmm`)

> SOTA multimodal machine-vision algorithms for manufacturing: defect detection, 6D pose / bin-picking, dimensional metrology, and predictive maintenance.

[![CI](https://github.com/CVKim/MachineVisionMultimodal/actions/workflows/ci.yml/badge.svg)](https://github.com/CVKim/MachineVisionMultimodal/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](environment.yml)

---

## Table of Contents

1. [What is this?](#1-what-is-this)
2. [Status matrix](#2-status-matrix)
3. [Repository layout](#3-repository-layout)
4. [Installation](#4-installation)
   - [4.1 Conda (recommended for GPU)](#41-conda-recommended-for-gpu)
   - [4.2 pip / venv](#42-pip--venv)
   - [4.3 Docker](#43-docker)
   - [4.4 Verifying the install](#44-verifying-the-install)
5. [Quickstart — the 5-minute smoke run](#5-quickstart--the-5-minute-smoke-run)
6. [Module guides](#6-module-guides)
   - [6.1 Anomaly detection — `mvmm.anomaly`](#61-anomaly-detection--mvmmanomaly)
   - [6.2 Dimensional metrology — `mvmm.metrology`](#62-dimensional-metrology--mvmmmetrology)
   - [6.3 6D pose & bin picking — `mvmm.pose`](#63-6d-pose--bin-picking--mvmmpose)
   - [6.4 Predictive maintenance — `mvmm.pdm`](#64-predictive-maintenance--mvmmpdm)
7. [CLI reference](#7-cli-reference)
8. [Configuration system (Hydra)](#8-configuration-system-hydra)
9. [Testing & continuous integration](#9-testing--continuous-integration)
10. [Branching & contribution workflow](#10-branching--contribution-workflow)
11. [Domain → algorithm mapping](#11-domain--algorithm-mapping)
12. [Roadmap & references](#12-roadmap--references)
13. [Troubleshooting / FAQ](#13-troubleshooting--faq)
14. [License](#14-license)

---

## 1. What is this?

`mvmm` is a **single repository, four pipelines** for applying 2024–2026
state-of-the-art computer-vision research to four problems that come up
constantly in real manufacturing lines:

| Problem | Module | Representative SOTA |
|---|---|---|
| Visual inspection / defect detection | [`mvmm.anomaly`](src/mvmm/anomaly) | PatchCore, EfficientAD, AnomalyCLIP, MultiADS, MuSc-V2 |
| Dimensional measurement | [`mvmm.metrology`](src/mvmm/metrology) | SAM2, Depth Anything v2, classical calibration |
| Bin picking / 6D pose | [`mvmm.pose`](src/mvmm/pose) | FoundationPose, SuperPose, Any6D, GraspNet |
| Predictive maintenance | [`mvmm.pdm`](src/mvmm/pdm) | TimesNet, PatchTST + vision fusion |

The repo is intentionally **hybrid** — every module exposes a classical
(rule-based) baseline next to its deep-learning counterpart, so you can
swap in whichever gives the better operating-point on your line.

### Why hybrid?

* Deep models hallucinate on out-of-distribution illumination.
* Rule-based pipelines miss subtle texture/pattern anomalies.
* Confidence-weighted fusion gives **both** explainability ("the rule
  fired because LBP rarity > τ") **and** recall (DL catches the rest).

---

## 2. Status matrix

| Module | Status | Notes |
|---|---|---|
| `anomaly.PatchCore` | ✅ Working baseline | Greedy coreset + optional FAISS |
| `anomaly.EfficientAD` | 🟡 Architecture only | Training loop is on the v0.2 roadmap |
| `anomaly.AnomalyCLIP` | ✅ Zero-shot inference | `open_clip` ViT-B/16 |
| `anomaly.hybrid` | ✅ Working | Canny + LBP + intensity z-score ensemble |
| `metrology.*` | ✅ Classical + SAM2 / DepthAnything wrappers | Wrappers lazy-import |
| `pose.PosePipeline` | ✅ Classical seg + antipodal grasps | FoundationPose plugged via env var |
| `pdm.MultimodalPdMModel` | ✅ Forward pass + CLI training | Data adapters are next |
| Docker / CI / tests | ✅ Pass on CPU runners | Smoke + module tests |

Legend: ✅ ready to use · 🟡 partially implemented · ⬜ planned

---

## 3. Repository layout

```
src/mvmm/
├── anomaly/            # PatchCore, EfficientAD, AnomalyCLIP, Hybrid rule+DL
│   ├── base.py
│   ├── patchcore.py
│   ├── efficient_ad.py
│   ├── anomaly_clip.py
│   └── hybrid.py
├── metrology/          # calibration, SAM2/GrabCut, Depth Anything, primitives
│   ├── calibration.py
│   ├── segmentation.py
│   ├── depth.py
│   └── measure.py
├── pose/               # FoundationPose wrapper, ICP, grasps, bin-pick pipeline
│   ├── icp_classical.py
│   ├── foundation_pose.py
│   ├── grasp.py
│   └── pipeline.py
├── pdm/                # TimesNet + Vision multimodal PdM
│   ├── timeseries.py
│   ├── vision.py
│   ├── fusion.py
│   └── datasets.py
├── common/             # dataset, transforms, metrics (AUROC/PRO), viz, IO
│   ├── data.py
│   ├── transforms.py
│   ├── metrics.py
│   ├── viz.py
│   └── io.py
└── cli.py              # Typer CLI: `mvmm <subcommand>`
configs/                # Hydra configs (anomaly / data / hardware)
scripts/                # download / make_sample_data / smoke_test / train_* / eval_*
tests/                  # pytest (CPU-only smoke + interface)
docs/                   # ROADMAP, ARCHITECTURE, papers.md
.github/                # CI, PR/Issue templates
data/sample/            # Committed synthetic dataset for CI/smoke
Dockerfile · docker-compose.yml · environment.yml · pyproject.toml
```

---

## 4. Installation

There are three supported paths. Pick one — they all give you the same
`mvmm` Python API and CLI.

### 4.1 Conda (recommended for GPU)

The conda recipe pins CUDA 12.1, PyTorch 2.x, FAISS-GPU and OpenCV — the
fastest path to a working RTX 30xx / 40xx environment.

```bash
conda env create -f environment.yml
conda activate mvmm
```

The recipe runs `pip install -e .` at the end, so `mvmm` is immediately
importable. If you only need parts of the stack:

```bash
conda env create -f environment.yml
conda activate mvmm
pip install -e ".[anomaly,viz]"     # subset extras
```

### 4.2 pip / venv

If you prefer a vanilla venv, install PyTorch **first** with the wheel
matching your CUDA, then the rest:

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\Activate.ps1
# Linux / macOS
source .venv/bin/activate

# PyTorch with CUDA 12.1 (skip --index-url for CPU)
pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch torchvision torchaudio

# Project + all extras
pip install -e ".[all]"
```

Available extras: `torch`, `anomaly`, `pose`, `pdm`, `viz`, `dev`, `all`.

### 4.3 Docker

Reproducible CUDA 12.1 image (~5 GB). Requires
[`nvidia-container-toolkit`](https://github.com/NVIDIA/nvidia-container-toolkit)
on the host.

```bash
docker compose build mvmm
docker compose run --rm mvmm mvmm info        # interactive smoke
docker compose run --rm mvmm python scripts/smoke_test.py
# Jupyter on http://localhost:8888 (no token)
docker compose up jupyter
```

Source / data / cache are bind-mounted, so edits on the host are
immediately visible in the container.

### 4.4 Verifying the install

```bash
mvmm info
```

Expected output:

```
mvmm version 0.1.0
  torch:       ok
  torchvision: ok
  open_clip:   ok
  open3d:      ok
  faiss:       ok
  cuda avail:  yes
   gpu[0]: NVIDIA GeForce RTX 3080
   gpu[1]: NVIDIA GeForce RTX 3080
```

Anything marked `missing` means the optional extra isn't installed; that
module's CLI commands will tell you exactly what to `pip install` when
you try them.

---

## 5. Quickstart — the 5-minute smoke run

This is the recommended first thing to run after install — no network,
no MVTec download, no API keys.

```bash
python scripts/smoke_test.py
```

What it does:

1. Generates a synthetic, MVTec-AD-shaped dataset under `data/sample/widget/`
2. Fits PatchCore (WideResNet50 backbone, ImageNet weights) in seconds
3. Evaluates on the held-out test split and prints image AUROC

Expected output:

```
[smoke] device = cuda
[smoke] fit done in 2.93s; bank size = torch.Size([1960, 1536])
[smoke] image AUROC = 1.0000
[smoke] PASS
```

If you see `[smoke] PASS`, everything is wired up correctly.

---

## 6. Module guides

### 6.1 Anomaly detection — `mvmm.anomaly`

#### PatchCore (working baseline)

```bash
# 1. Download MVTec-AD (one-time, ~5 GB)
python scripts/download_mvtec.py --dest data/mvtec_ad

# 2. Fit a memory bank on a single category
mvmm anomaly train \
    --data-root data/mvtec_ad/bottle \
    --output    checkpoints/patchcore_bottle.pkl

# 3. Evaluate
mvmm anomaly eval \
    --data-root  data/mvtec_ad/bottle \
    --checkpoint checkpoints/patchcore_bottle.pkl
```

Typical numbers on MVTec-AD `bottle` with WideResNet50 + 10% coreset:
**image AUROC > 0.98, pixel AUROC > 0.97**.

#### Zero-shot with CLIP (no training)

```bash
mvmm anomaly zero-shot \
    --image       data/sample/widget/test/defect/000.png \
    --object-name "industrial widget"
# → outputs/clip_score.png  (heatmap overlay)
```

This uses the WinCLIP-style multi-scale window scoring with handcrafted
prompts. Replace `--object-name` with any noun phrase that describes
your part ("automotive bracket", "lithium-ion cell can lid", etc.).

#### Rule + DL fusion (Python API)

```python
from mvmm.anomaly.hybrid import RuleBasedScorer, fuse_rule_and_dl
from mvmm.anomaly.patchcore import PatchCore

rule = RuleBasedScorer().score(image_rgb)              # (H, W) float
dl   = patchcore.predict(image_tensor.unsqueeze(0))    # AnomalyResult
fused = fuse_rule_and_dl(rule, dl, alpha=0.6)          # AnomalyResult
```

### 6.2 Dimensional metrology — `mvmm.metrology`

#### One-shot measurement on a single image

```bash
mvmm metrology measure \
    --image     data/your_part.png \
    --seed-x    320 \
    --seed-y    240 \
    --mm-per-px 0.12         # set 0 to report pixels only
```

Output (JSON written to `outputs/measure.json`):

```json
{
  "rect_width_px":   180.2,
  "rect_height_px":  301.7,
  "rect_width_mm":    21.62,
  "rect_height_mm":   36.20,
  "circle_radius_px": 165.4,
  "circle_radius_mm": 19.85,
  "line_length_px":  290.5,
  "line_length_mm":   34.86
}
```

The default segmenter is GrabCut (no GPU). To switch to SAM2:

```python
from mvmm.metrology.segmentation import build_segmenter
seg = build_segmenter(backend="sam2",
                      checkpoint="path/to/sam2_hiera_large.pt")
mask = seg(image_rgb, points=[(cx, cy, 1)])
```

#### Camera calibration helper

```python
from mvmm.metrology.calibration import calibrate_with_chessboard

intr, rms = calibrate_with_chessboard(
    image_paths=glob("calib_*.png"),
    pattern_size=(9, 6),
    square_mm=25.0,
)
print(f"RMS reproj. error: {rms:.3f} px")
intr.save_npz("data/intrinsics.npz")
```

### 6.3 6D pose & bin picking — `mvmm.pose`

#### Run a single bin-pick frame end-to-end (classical baseline)

```bash
mvmm pose bin-pick \
    --rgb            data/scene.png \
    --depth          data/scene_depth.png \
    --intrinsics-npz data/intrinsics.npz \
    --seed-x         640 \
    --seed-y         360
```

The classical pipeline uses GrabCut for segmentation, projects the
masked depth into 3D, then runs the antipodal grasp sampler. Output
(`outputs/binpick.json`) contains the best grasp's two contact points,
opening width in mm, and a force-closure proxy score.

#### Adding FoundationPose (model-aware refinement)

1. Clone NVlabs/FoundationPose into `third_party/` and install per its README.
2. Set `MVMM_FOUNDATIONPOSE_PATH` to the cloned repo's root.
3. Use `mvmm.pose.foundation_pose.FoundationPoseEstimator` in your pipeline:

```python
from mvmm.pose.foundation_pose import FoundationPoseEstimator
from mvmm.pose.pipeline import PosePipeline
from mvmm.pose.grasp import antipodal_grasps
from mvmm.metrology.segmentation import build_segmenter

fp = FoundationPoseEstimator(
    cad_mesh_path="data/cad/part.obj",
    intrinsics_3x3=intr.K,
)
pipeline = PosePipeline(
    segmenter=build_segmenter("sam2", checkpoint="..."),
    pose_estimator=fp,
    grasp_sampler=antipodal_grasps,
)
```

### 6.4 Predictive maintenance — `mvmm.pdm`

#### Train the multimodal model

Expected table layout (`csv` or `parquet`):

```
timestamp, vib_x, vib_y, current, ..., health, image_path
1700000000, 0.12, -0.04, 1.83, ..., 0, frames/000001.png
1700000001, 0.13, -0.05, 1.85, ..., 0, frames/000002.png
...
```

Train:

```bash
mvmm pdm train \
    --table       data/motor.csv \
    --sensor-cols "vib_x,vib_y,current" \
    --target-col  health \
    --seq-len     512 \
    --epochs      5
```

The sensor branch is a small TimesNet stack; the image branch is a
frozen `torchvision` ResNet (or any timm model — change in
`PdMConfig.vision_backbone`). Late-fusion MLP produces classification
logits or a scalar RUL.

---

## 7. CLI reference

The top-level command is `mvmm`. All subcommands print help with `--help`.

```
mvmm                              ─ overview
├── info                          ─ environment / dependency check
├── anomaly
│   ├── train                     ─ fit PatchCore on a category
│   ├── eval                      ─ AUROC / PRO on test split
│   └── zero-shot                 ─ AnomalyCLIP inference on one image
├── metrology
│   └── measure                   ─ segment + measure dimensions
├── pose
│   └── bin-pick                  ─ classical bin-pick on RGB+depth
└── pdm
    └── train                     ─ train TimesNet + Vision PdM model
```

---

## 8. Configuration system (Hydra)

The CLI exposes flat flags for quick iteration; the scripts under
`scripts/` use [Hydra](https://hydra.cc/) for reproducible experiments.

```
configs/
├── defaults.yaml
├── anomaly/
│   ├── patchcore.yaml
│   ├── efficient_ad.yaml
│   └── anomaly_clip.yaml
├── data/
│   ├── mvtec_ad.yaml
│   └── visa.yaml
└── hardware/
    ├── cuda.yaml
    └── cpu.yaml
```

Example overrides:

```bash
python scripts/train_anomaly.py \
    anomaly=patchcore \
    anomaly.coreset_ratio=0.05 \
    data=mvtec_ad data.category=cable \
    hardware=cuda
```

Hydra writes a per-run output directory under `outputs/<date>/<time>/`
with the resolved config + logs.

---

## 9. Testing & continuous integration

Local:

```bash
ruff check src tests scripts
ruff format src tests scripts
pytest -m "not slow and not gpu and not network"
python scripts/smoke_test.py
```

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs the
same set on Ubuntu, Python 3.10 and 3.11, CPU-only PyTorch. Every PR
must be green before merging into `dev`.

Test markers:
- `slow`     — runs >5 s
- `gpu`      — requires CUDA
- `network`  — downloads from the internet

The default `pytest` invocation skips all three.

---

## 10. Branching & contribution workflow

```
main      ←  always stable / releasable. No direct commits.
└─ dev    ←  integration. All feature PRs target this.
   └─ feature/<topic>   ←  actual work
```

Typical loop:

```bash
git checkout dev && git pull
git checkout -b feature/dinov2-patchcore
# code, commit
ruff check src tests scripts && pytest
git push -u origin feature/dinov2-patchcore
gh pr create --base dev --title "feat(anomaly): DINOv2 backbone for PatchCore"
# ... review + CI green ...
# Merge feature → dev. Release time: PR dev → main.
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for commit conventions and
pre-commit hook setup.

---

## 11. Domain → algorithm mapping

A pragmatic cheat-sheet for picking a starting module given a real
manufacturing task.

| Your task | Start here | Upgrade path |
|---|---|---|
| Wafer-level pattern defect | `anomaly.PatchCore` (WideResNet50) | EfficientAD for real-time → AnomalyCLIP for new patterns |
| Automotive surface scratch / dent | `anomaly.hybrid` (Canny + LBP + PatchCore) | Swap to DINOv2 / ConvNeXt-v2 backbones |
| Diameter / length / flatness | `metrology.measure` + SAM2 | Stereo (`StereoSGBM`) or Depth Anything v2 |
| Single-object bin pick | `pose.PosePipeline` + antipodal grasps | FoundationPose + SAM2 (SuperPose-style) |
| Motor / pump health | `pdm.MultimodalPdMModel` (TimesNet + ResNet) | Add PatchTST, TimeLLM |

---

## 12. Roadmap & references

- [docs/ROADMAP.md](docs/ROADMAP.md) — versioned milestone plan
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — design principles &
  how to add a new SOTA method
- [docs/papers.md](docs/papers.md) — curated SOTA reference list with
  module mappings (2022–2026)

---

## 13. Troubleshooting / FAQ

**`ImportError: cannot import name 'WideResNet50_2_Weights' …`**
Your torchvision is newer than 0.18 — the symbol is `Wide_ResNet50_2_Weights`.
This repo already uses the underscore form; if you see this error in
your own code, follow the same convention.

**`KeyError: 'mask'` during DataLoader collate**
The `MVTecADDataset` was returning samples without a `"mask"` key for
"good" test images. Already fixed; if you see it, pull `dev`.

**`UnicodeEncodeError: 'cp949' codec can't encode character ...`**
Windows console using CP949 codec. Either set
`PYTHONIOENCODING=utf-8` in your shell, or upgrade to the latest
PowerShell (it defaults to UTF-8).

**`mvmm anomaly zero-shot` is slow on CPU**
Expected — `open_clip` ViT-B/16 is ~150 M params. Use `--device cpu`
only for sanity; switch to a CUDA device for any real volume.

**FAISS install fails on Windows**
`faiss-gpu` is Linux-only. On Windows, `faiss-cpu` works fine for the
PatchCore memory-bank sizes used here. The code auto-falls back to
`torch.cdist` if FAISS is unavailable.

**Docker: `nvidia-container-cli: initialization error`**
Ensure NVIDIA Container Toolkit is installed and Docker has been
restarted after install. On WSL2, also enable GPU support in Docker
Desktop → Settings → Resources → WSL Integration.

---

## 14. License

[MIT](LICENSE) © 2026 CVKim
