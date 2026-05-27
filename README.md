# mvmm — Multimodal Perception Stack for Industrial CCTV & 3D

> Production-grade reference implementations of the 2024–2026 SOTA stack for
> **CCTV / video tracking · zero-shot perception · 3D scene understanding ·
> predictive maintenance + video anomaly detection**, with a single CLI and
> a one-shot inference dump.

[![CI](https://github.com/CVKim/MachineVisionMultimodal/actions/workflows/ci.yml/badge.svg)](https://github.com/CVKim/MachineVisionMultimodal/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](environment.yml)

---

## Table of Contents

1. [Why this repo exists](#1-why-this-repo-exists)
2. [The four pillars (and their SOTA)](#2-the-four-pillars-and-their-sota)
3. [Repository layout](#3-repository-layout)
4. [Installation](#4-installation)
   - [4.1 Conda (recommended)](#41-conda-recommended)
   - [4.2 pip / venv](#42-pip--venv)
   - [4.3 Docker](#43-docker)
   - [4.4 Verifying the install](#44-verifying-the-install)
5. [The 5-command tour](#5-the-5-command-tour)
6. [Pillar 1 — CCTV tracking](#6-pillar-1--cctv-tracking)
7. [Pillar 2 — Zero-shot / open-vocabulary perception](#7-pillar-2--zero-shot--open-vocabulary-perception)
8. [Pillar 3 — 3D scene understanding](#8-pillar-3--3d-scene-understanding)
9. [Pillar 4 — PdM + Video Anomaly Detection](#9-pillar-4--pdm--video-anomaly-detection)
10. [One-shot inference dump](#10-one-shot-inference-dump)
11. [Gradio demo](#11-gradio-demo)
12. [CLI reference](#12-cli-reference)
13. [Testing & CI](#13-testing--ci)
14. [Branching & contribution workflow](#14-branching--contribution-workflow)
15. [Roadmap & references](#15-roadmap--references)
16. [Troubleshooting / FAQ](#16-troubleshooting--faq)
17. [License](#17-license)

---

## 1. Why this repo exists

This is a curated, runnable reference stack for the four most career-relevant
problems in industrial CV in 2025–2026:

| Pillar | Real-world impact |
|---|---|
| **CCTV tracking** | Worker safety zones, AGV / forklift / pallet tracking, line monitoring, behavior analytics |
| **Zero-shot perception** | New-SKU first-day inspection, text-prompted detection on unlabeled cameras |
| **3D scene understanding** | Depth, 6D pose for bin picking, point clouds, digital-twin assets via Gaussian splatting |
| **PdM + Video Anomaly** | Equipment health from sensors + camera, abnormal-event detection on CCTV |

Each pillar ships:
- A working baseline you can run today
- Wrappers for the current SOTA models (lazy-imported)
- A CLI subcommand and demo script
- Tests covering the pure-Python logic (CPU-only, fast)

The legacy manufacturing modules (PatchCore image AD, MVTec metrology,
6D-pose bin pick) are still here under `three_d/` and `anomaly/` —
nothing was deleted, only re-organized.

---

## 2. The four pillars (and their SOTA)

### CCTV tracking — `mvmm.tracking`
| Component | Implementation | SOTA reference |
|---|---|---|
| Detector | YOLOv8/v11, RT-DETR via `ultralytics` | YOLO-World, RT-DETR |
| Open-vocab detector | GroundingDINO via HF transformers | Grounding DINO 1.5 |
| Tracker | ByteTrack via `supervision` | ByteTrack, BoT-SORT, MASA |
| Video segmentation | SAM2 video predictor wrapper | SAM2 (Meta, 2024) |
| Appearance ReID | CLIP image embeddings + OSNet | TransReID 2024 line |
| Analytics | Polygon zones, line crossing, dwell timers | — |

### Zero-shot — `mvmm.zeroshot`
| Task | Implementation | SOTA reference |
|---|---|---|
| Image anomaly | `AnomalyCLIP` (WinCLIP-style windowed) | AnomalyCLIP (ICLR'24), MuSc-V2 |
| Open-vocab detection | GroundingDINO + OWLv2 | OWLv2, GDINO 1.5 |
| Promptable segmentation | SAM2 image predictor wrapper | SAM2 |
| Detect → segment pipeline | `OpenVocabPipeline` | Grounded-SAM-2 |

### 3D — `mvmm.three_d`
| Task | Implementation | SOTA reference |
|---|---|---|
| Monocular depth | Depth Anything v2 (HF transformers) | DAv2, MoGe, Marigold |
| Stereo depth | OpenCV SGBM | — |
| 6D pose (CAD-based) | FoundationPose wrapper | FoundationPose (CVPR'24 highlight) |
| 6D pose (CAD-free) | Any6D wrapper | Any6D (CVPR'25) |
| Scene reconstruction | depth → point cloud → PLY | — |
| Gaussian splatting | `gsplat` backend wrapper | 3DGS (SIGGRAPH'23) |
| Grasping | Antipodal sampler | AnyGrasp |
| Metrology | Calibration · GrabCut/SAM2 segmentation · circle/line/rect fit | — |

### PdM + VAD — `mvmm.pdm`, `mvmm.vad`
| Task | Implementation | SOTA reference |
|---|---|---|
| Multivariate time-series | TimesNet | TimesNet, PatchTST, TimeLLM |
| Multimodal PdM | TimesNet + frozen ResNet fusion (concat MLP) | — |
| Video anomaly | Frame autoencoder + MemAE | MemAE, MGFN, MULDE |
| VAD dataset adapter | Flat & nested image-folder layouts | UCF-Crime, ShanghaiTech |

---

## 3. Repository layout

```
src/mvmm/
├── tracking/           # CCTV tracking pipeline
│   ├── detectors.py        # YOLO, GroundingDINO
│   ├── byte_track.py       # ByteTrack online tracker
│   ├── sam2_video.py       # SAM2 video predictor wrapper
│   ├── reid.py             # CLIP / OSNet appearance ReID
│   ├── analytics.py        # zones, line counters, dwell timers
│   └── pipeline.py         # end-to-end detect → track → annotate
├── zeroshot/           # Open-vocabulary perception
│   ├── grounding_dino.py
│   ├── owl_v2.py
│   ├── sam2_promptable.py
│   ├── anomaly_clip.py     # image-level zero-shot AD
│   └── pipeline.py         # text → boxes → masks
├── three_d/            # 3D perception
│   ├── depth/              # DepthAnythingV2, MoGe, Marigold, SGBM
│   ├── pose/               # FoundationPose, Any6D, ICP, grasps
│   ├── metrology/          # calibration, segmentation, measurement
│   ├── reconstruction.py   # depth → point cloud, PLY export
│   └── gaussian_splatting.py  # gsplat wrapper
├── vad/                # Video anomaly detection
│   ├── conv_autoencoder.py
│   ├── memae.py
│   └── datasets.py
├── pdm/                # Predictive maintenance
│   ├── timeseries.py       # TimesNet
│   ├── vision.py           # frozen image backbone
│   ├── fusion.py           # multimodal classifier / RUL
│   └── datasets.py
├── anomaly/            # Image-level AD (PatchCore + hybrid)
├── common/             # data, transforms, metrics, viz, io
├── serving/            # Gradio app
└── cli.py              # `mvmm <subcommand>`
configs/                # Hydra configs
scripts/                # train_* / eval_* / make_sample_* / dump_all_results
tests/                  # pytest (CPU-only, no network)
docs/                   # ROADMAP, ARCHITECTURE, papers.md
data/sample/            # committed synthetic datasets (widget AD, motor PdM, vad, cctv)
```

---

## 4. Installation

### 4.1 Conda (recommended)

```bash
conda env create -f environment.yml
conda activate mvmm
pip install ultralytics supervision    # tracking extras
```

### 4.2 pip / venv

```bash
python -m venv .venv
.venv\Scripts\activate.bat              # cmd.exe
# .\.venv\Scripts\Activate.ps1          # PowerShell

pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch torchvision torchaudio
pip install -e ".[all]" ultralytics supervision
```

### 4.3 Docker

```bash
docker compose build mvmm
docker compose run --rm mvmm mvmm info
docker compose up jupyter    # http://localhost:8888
```

### 4.4 Verifying the install

```cmd
mvmm info
```

Expected output (your environment may show some libs as missing — those
modules will surface a clear install message when you call them):

```
mvmm version 0.1.0
  torch:        ok
  torchvision:  ok
  open_clip:    ok
  open3d:       missing
  faiss:        ok
  ultralytics:  ok    [for tracking]
  supervision:  ok    [for ByteTrack]
  transformers: ok    [for GDINO/OWLv2]
  sam2:         missing [for SAM2]
  gsplat:       missing [for 3D Gaussian Splatting]
  cuda avail:   yes
   gpu[0]: NVIDIA GeForce RTX 3080
   gpu[1]: NVIDIA GeForce RTX 3080
```

---

## 5. The 5-command tour

Each of these is < 30 seconds on an RTX 3080. They use only the
committed synthetic samples — no downloads, no internet.

```cmd
:: 1. Sanity (PatchCore image AD on synthetic widget data)
python scripts\smoke_test.py

:: 2. Zero-shot CLIP prompt comparison
python scripts\demo_zero_shot_prompts.py

:: 3. PdM train + eval on synthetic motor data
mvmm pdm train --table data\sample\pdm\motor.csv ^
    --sensor-cols "vib_x,vib_y,current" --target-col health ^
    --seq-len 512 --stride 512 --epochs 5 --batch-size 16 ^
    --output checkpoints\pdm_motor.pt
mvmm pdm eval --table data\sample\pdm\motor.csv ^
    --sensor-cols "vib_x,vib_y,current" --target-col health ^
    --checkpoint checkpoints\pdm_motor.pt --seq-len 512 --stride 512

:: 4. VAD train + eval on synthetic CCTV anomaly data
python scripts\make_vad_sample_data.py
mvmm vad train --train-root data\sample\vad\train\normal --epochs 15 ^
    --output checkpoints\vad_ae.pt
mvmm vad eval --test-root data\sample\vad\test ^
    --labels data\sample\vad\labels.csv --checkpoint checkpoints\vad_ae.pt

:: 5. CCTV tracking end-to-end on a synthetic clip
python scripts\make_track_sample_video.py
mvmm track video --input data\sample\track\sample.mp4 ^
    --output-video outputs\track_sample.mp4 ^
    --output-json  outputs\track_sample.json --model yolov8n.pt
```

---

## 6. Pillar 1 — CCTV tracking

### What the pipeline does

```
RTSP / mp4 ─▶ frame loop ─▶ detector ─▶ ByteTrack ─▶ analytics ─▶ annotated mp4 + per-frame JSON
                                                          │
                                                          ├─ Zone (count people inside a polygon)
                                                          ├─ Line counter (entry / exit)
                                                          └─ Dwell timer (per-track seconds in zone)
```

### Closed-vocabulary (fast path)

```cmd
mvmm track video --input cctv\cam01_20260527.mp4 ^
    --output-video outputs\cam01_track.mp4 ^
    --output-json  outputs\cam01_track.json ^
    --detector yolo --model yolo11s.pt --classes "person,forklift" ^
    --score-threshold 0.30 --frame-rate 25
```

### Open-vocabulary (text-prompted detection)

```cmd
mvmm track video --input cctv\cam01.mp4 --detector grounding_dino ^
    --classes "person wearing safety vest,forklift,pallet,box on the floor"
```

### Python API — building a richer pipeline

```python
import numpy as np
from mvmm.tracking import ByteTrackTracker, TrackingPipeline, build_detector
from mvmm.tracking.analytics import PolygonZone, LineCounter, DwellTimer

detector = build_detector("yolo", model="yolo11s.pt", device="cuda")
tracker  = ByteTrackTracker(frame_rate=25, track_activation_threshold=0.3)

safe_zone  = PolygonZone(np.array([[100, 200], [500, 200], [500, 460], [100, 460]]), name="safe")
entry_line = LineCounter(a=(0, 250), b=(640, 250), name="entry")
dwell      = DwellTimer(zone=safe_zone)

pipeline = TrackingPipeline(
    detector=detector, tracker=tracker,
    classes=["person", "forklift"], score_threshold=0.3,
    zones=[safe_zone], line_counters=[entry_line], dwell_timers=[dwell],
)
stats = pipeline.process_video("cam01.mp4",
                               output_video="outputs/cam01.mp4",
                               output_json="outputs/cam01.json")
print(f"{stats.n_unique_track_ids} unique IDs; entry={entry_line.in_count}")
```

### Optional: SAM2 video mask tracking

```python
from mvmm.tracking.sam2_video import SAM2VideoTracker
seg = SAM2VideoTracker(checkpoint="weights/sam2_hiera_l.pt", device="cuda")
seg.init_state("cam01.mp4")
seg.add_box_prompts(frame_idx=0, boxes_xyxy=first_frame_boxes, obj_ids=[1, 2, 3])
masks = seg.propagate()  # dict[frame_idx][obj_id] = HxW uint8 mask
```

---

## 7. Pillar 2 — Zero-shot / open-vocabulary perception

### Open-vocabulary detection on one image

```cmd
mvmm zeroshot detect ^
    --image data\sample\widget\test\defect\000.png ^
    --classes "person,forklift,pallet,defect on the surface" ^
    --detector grounding_dino ^
    --output outputs\zeroshot_detect.png
```

### Zero-shot image anomaly detection (no training data)

```cmd
mvmm anomaly zero-shot ^
    --image data\sample\widget\test\defect\000.png ^
    --object-name "industrial widget" ^
    --output outputs\clip_score.png
```

Prompt engineering matters — see `scripts\demo_zero_shot_prompts.py`
for a head-to-head comparison of generic vs domain-specific prompt sets:

| Prompt set | normal | defect | gap |
|---|---|---|---|
| generic | 0.9265 | 0.9256 | ≈ 0 (bad) |
| semiconductor | 0.3778 | 0.5387 | +0.16 (good) |
| automotive | 0.8810 | 0.9031 | +0.02 |

### Detect → segment in one call

```python
from mvmm.tracking.detectors import GroundingDINODetector
from mvmm.zeroshot import SAM2ImagePredictor, OpenVocabPipeline

det = GroundingDINODetector()
seg = SAM2ImagePredictor(checkpoint="weights/sam2_hiera_l.pt")
pipeline = OpenVocabPipeline(detector=det, segmenter=seg)

result = pipeline(image_rgb, classes=["forklift", "pallet"])
# result.detections.boxes, result.masks
```

---

## 8. Pillar 3 — 3D scene understanding

### Monocular depth → PLY point cloud

```cmd
mvmm depth infer ^
    --image    data\sample\widget\test\defect\000.png ^
    --output   outputs\depth.png ^
    --output-ply outputs\depth.ply
```

```
Depth saved outputs\depth.png  (min=2.129, max=4.145)
PLY saved:  outputs\depth.ply  (65536 points)
```

### Programmatic depth + reconstruction

```python
import numpy as np
from mvmm.three_d.depth import DepthAnythingV2
from mvmm.three_d.reconstruction import back_project, save_ply

depth = DepthAnythingV2(device="cuda")(image_rgb)
h, w = depth.shape
K = np.array([[800, 0, w / 2], [0, 800, h / 2], [0, 0, 1]])
out = back_project(depth, K, rgb=image_rgb)
save_ply("scene.ply", out["points"], out["colors"])
```

### 6D pose (CAD-based, requires FoundationPose install)

```python
from mvmm.three_d.pose.foundation_pose import FoundationPoseEstimator
fp = FoundationPoseEstimator(cad_mesh_path="cad/part.obj",
                              intrinsics_3x3=K)
pose = fp.estimate(rgb=image_rgb, depth_mm=depth_mm, mask=part_mask)
```

### Gaussian Splatting (requires `pip install gsplat`)

```python
from mvmm.three_d.gaussian_splatting import GaussianSplattingTrainer
trainer = GaussianSplattingTrainer(backend="gsplat")
trainer.train(scene_dir="scenes/factory_floor", output_dir="outputs/gs", iterations=7000)
```

---

## 9. Pillar 4 — PdM + Video Anomaly Detection

### PdM — train + eval on synthetic motor data

```cmd
python scripts\make_pdm_sample_data.py
mvmm pdm train --table data\sample\pdm\motor.csv ^
    --sensor-cols "vib_x,vib_y,current" --target-col health ^
    --seq-len 512 --stride 512 --epochs 5 --output checkpoints\pdm_motor.pt
mvmm pdm eval --table data\sample\pdm\motor.csv ^
    --sensor-cols "vib_x,vib_y,current" --target-col health ^
    --checkpoint checkpoints\pdm_motor.pt --seq-len 512 --stride 512
```

Observed on the synthetic motor (5 epochs, 100 windows):

```
loss 0.66 → 0.32
PdM eval  windows=100  accuracy=0.9600  AUROC=0.9984
```

### VAD — train a frame autoencoder, evaluate on a mixed test set

```cmd
python scripts\make_vad_sample_data.py
mvmm vad train --train-root data\sample\vad\train\normal --epochs 15 ^
    --output checkpoints\vad_ae.pt
mvmm vad eval --test-root data\sample\vad\test ^
    --labels data\sample\vad\labels.csv --checkpoint checkpoints\vad_ae.pt
```

Observed:

```
recon_loss 0.0215 → 0.00065 (after 15 epochs)
VAD eval  frames=160  AUROC=0.7990
  test_anomaly_00   mean=0.01706  max=0.04225
  test_anomaly_01   mean=0.01706  max=0.04224
  test_normal_00    mean=0.00067  max=0.00069
  test_normal_01    mean=0.00067  max=0.00069
```

Normal clips reconstruct ~25× better than anomalous ones.

### MemAE upgrade

Swap `ConvAutoEncoder` for `MemAE` to harden the gap on harder datasets:

```python
from mvmm.vad.memae import MemAE, entropy_loss
model = MemAE(in_channels=3, n_slots=2000)
# loss = MSE(recon, x) + 0.0002 * entropy_loss(attention)
```

---

## 10. One-shot inference dump

```cmd
python scripts\dump_all_results.py --inputs data\sample --out outputs\dump
```

What it does: for every image/video under `--inputs`, runs depth +
zero-shot detection + AnomalyCLIP (for images) and tracking (for
videos), writes everything under `outputs\dump\<timestamp>\` with a
``summary.json`` recording which modules succeeded and how long they
took.

```
outputs/dump/20260527_103454/
├── images/
│   ├── 000__depth.png
│   ├── 000__zeroshot.png
│   └── 000__clip.png
├── videos/
│   ├── sample__track.mp4
│   └── sample__track.json
└── summary.json
```

---

## 11. Gradio demo

```cmd
python -m mvmm.serving.gradio_app
# → http://localhost:7860
```

Three tabs:
- **CCTV Tracking** — upload a clip, set classes, see the annotated mp4
- **Zero-shot Detection** — upload an image, type text classes, pick GroundingDINO vs OWLv2
- **Monocular Depth** — upload an image, get a colorized Depth Anything v2 map

---

## 12. CLI reference

```
mvmm
├── info                              ─ environment / dep check
├── track
│   └── video                          ─ end-to-end CCTV tracking
├── zeroshot
│   └── detect                         ─ open-vocab detection on an image
├── depth
│   └── infer                          ─ monocular depth + optional PLY
├── vad
│   ├── train                          ─ frame-AE training on normal frames
│   └── eval                           ─ AUROC + per-clip scores
├── anomaly
│   ├── train / eval / zero-shot       ─ image AD (PatchCore / AnomalyCLIP)
├── metrology
│   └── measure                        ─ segment + dimensions
├── pose
│   └── bin-pick                       ─ classical bin-pick on RGB+D
└── pdm
    ├── train / eval
```

Every subcommand prints help with `--help`.

---

## 13. Testing & CI

Local:

```cmd
ruff check src tests scripts
ruff format src tests scripts
set PYTHONPATH=src && pytest -v
python scripts\smoke_test.py
```

Current state: **29 tests passing** (CPU-only, no network), covering
common utilities, image AD, metrology, pose, PdM, VAD, tracking
analytics, 3D reconstruction. CI runs the same on Ubuntu, Python 3.10
and 3.11.

---

## 14. Branching & contribution workflow

```
main      ←  stable / releasable. No direct commits.
└─ dev    ←  integration. All feature PRs target this.
   └─ feature/<topic>   ←  actual work
```

```cmd
git checkout dev
git pull
git checkout -b feature/your-topic
:: code, ruff, pytest
git push -u origin feature/your-topic
gh pr create --base dev
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for commit message conventions and pre-commit hooks.

---

## 15. Roadmap & references

- [docs/ROADMAP.md](docs/ROADMAP.md) — versioned milestone plan
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — design principles + how to add a new SOTA method
- [docs/papers.md](docs/papers.md) — curated SOTA reference list with module mappings (2022–2026)

Next planned milestones (see ROADMAP):
- **v0.2** — Cross-camera ReID (Market-1501 + TransReID), DEVA / MASA tracker swap-ins
- **v0.3** — AnomalyCLIP learnable prompts, GroundingDINO 1.5, Grounded-SAM-2
- **v0.4** — FoundationPose container, 3DGS end-to-end notebook, MoGe / Marigold
- **v0.5** — MemAE / MGFN VAD training on UCF-Crime, PatchTST + cross-attention for PdM

---

## 16. Troubleshooting / FAQ

**`$env:PYTHONPATH=...` says "syntax error" on Windows**
You're in `cmd.exe`, not PowerShell. Use `set PYTHONPATH=src && ...`.

**`mvmm track video` returns 0 detections**
YOLO is trained on COCO — if your objects aren't in COCO classes,
switch to `--detector grounding_dino` and supply text classes.

**FAISS install fails on Windows**
`faiss-gpu` is Linux-only. PatchCore auto-falls back to `torch.cdist`
when FAISS is unavailable.

**`ImportError: sam2 is not installed`**
SAM2 must be installed from the upstream repo:
`pip install "git+https://github.com/facebookresearch/sam2"`.
Until then, the SAM2 wrappers raise a clear ImportError when called.

**Tracking inference is slow**
Drop to `yolov8n.pt` for the fastest baseline, or run with
`--device cpu` and a smaller `--frame-rate` for quick iteration.

**Depth Anything v2 download fails behind a proxy**
Set `HF_ENDPOINT=https://hf-mirror.com` (or your enterprise mirror)
before running the depth command.

---

## 17. License

[MIT](LICENSE) © 2026 CVKim
