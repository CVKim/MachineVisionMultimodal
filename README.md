# MachineVisionMultimodal (mvmm)

> 제조 도메인을 위한 **SOTA 멀티모달 머신비전** 알고리즘 모노레포
> SOTA multimodal machine-vision algorithms for manufacturing — defect detection, 6D pose / bin-picking, dimensional metrology, and predictive maintenance.

[![CI](https://github.com/CVKim/MachineVisionMultimodal/actions/workflows/ci.yml/badge.svg)](https://github.com/CVKim/MachineVisionMultimodal/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](environment.yml)

---

## 한국어 요약

본 레포는 **반도체·자동차 부품 제조** 현장에서 가장 자주 부딪히는 네 가지 문제를
2025–2026년 SOTA로 다루는 학습/포트폴리오용 모노레포입니다. 룰베이스(1~3D 고전 비전)
강점을 그대로 가져가면서, 그 위에 **딥러닝 + Vision-Language + 멀티모달 융합**을
얹어 실제 산업 데이터에 바로 붙일 수 있도록 설계했습니다.

| 문제 | 본 레포 모듈 | 대표 SOTA |
|---|---|---|
| 외관 검사 / 불량 탐지 | `mvmm.anomaly` | PatchCore, EfficientAD, AnomalyCLIP, MultiADS, MuSc-V2 |
| 치수 측정 | `mvmm.metrology` | SAM2, Depth Anything v2, 전통 캘리브레이션 |
| 빈피킹 / 6D 포즈 | `mvmm.pose` | FoundationPose, SuperPose, Any6D, GraspNet |
| 설비 예지보전 (PdM) | `mvmm.pdm` | TimesNet, PatchTST + 비전 융합 |

---

## ✨ Highlights

- **One install, four pipelines.** `pip install -e ".[all]"` → `mvmm info` → `mvmm anomaly ...`
- **Hybrid rule + DL.** 룰베이스 신호(`hybrid.py`)와 DL 점수를 가중 융합 — 설명가능성 + 재현율.
- **Zero-shot ready.** AnomalyCLIP (WinCLIP/AnomalyCLIP 계열)로 신규 SKU 첫날부터 검사 가능.
- **3D & multimodal.** SAM2 + Depth Anything v2 → 마스크 → 픽셀→mm 변환까지 한 줄.
- **Reproducible.** Conda / Docker / pip 어디서나 동일 결과. CI는 GitHub Actions로 모든 PR 검증.
- **Branch discipline.** `main`(stable) ⇽ `dev`(integration) ⇽ `feature/*` — PR 템플릿 포함.

---

## 🚀 Quickstart (한국어)

### 1. 환경 구성 — 셋 중 하나만

```powershell
# (a) Conda — 권장 (GPU/FAISS 모두 자동)
conda env create -f environment.yml
conda activate mvmm

# (b) pip — 가볍게
python -m venv .venv
.\.venv\Scripts\activate
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
pip install -e ".[all]"

# (c) Docker — 어디서나 동일
docker compose build mvmm
docker compose run --rm mvmm mvmm info
```

### 2. 동작 확인 (네트워크 없이 30초)

```powershell
python scripts/smoke_test.py
# → 합성 위젯 이미지 생성 → PatchCore fit → image AUROC > 0.7 → PASS ✔
```

### 3. 실제 MVTec-AD에서 학습 → 평가

```powershell
python scripts/download_mvtec.py --dest data/mvtec_ad
mvmm anomaly train --data-root data/mvtec_ad/bottle --output checkpoints/patchcore_bottle.pkl
mvmm anomaly eval  --data-root data/mvtec_ad/bottle --checkpoint checkpoints/patchcore_bottle.pkl
# 일반적으로 PatchCore + WideResNet50로 image AUROC > 0.98 (bottle)
```

### 4. 새 SKU에 학습 데이터 없이 (Zero-shot)

```powershell
mvmm anomaly zero-shot --image data/your_part.png --object-name "automotive bracket"
# → outputs/clip_score.png 에 히트맵 저장
```

### 5. 치수 측정 한 번

```powershell
mvmm metrology measure --image data/your_part.png --seed-x 320 --seed-y 240 --mm-per-px 0.12
```

### 6. 빈피킹 한 프레임

```powershell
mvmm pose bin-pick --rgb data/scene.png --depth data/scene_depth.png --intrinsics-npz data/intr.npz
```

### 7. 설비 예지보전 (시계열 + 영상)

```powershell
mvmm pdm train --table data/motor.csv --sensor-cols "vib_x,vib_y,current" --target-col health --epochs 5
```

---

## 🇺🇸 English Quickstart

```bash
conda env create -f environment.yml && conda activate mvmm
python scripts/smoke_test.py                       # 30s sanity
mvmm info                                          # show env
mvmm anomaly train --data-root data/mvtec_ad/bottle --output ckpt.pkl
mvmm anomaly eval  --data-root data/mvtec_ad/bottle --checkpoint ckpt.pkl
```

---

## 🧭 Repo Layout

```
src/mvmm/
├── anomaly/            # PatchCore, EfficientAD, AnomalyCLIP, Hybrid 룰+DL
├── metrology/          # 캘리브, SAM2/GrabCut 세그멘테이션, Depth Anything, 측정 프리미티브
├── pose/               # FoundationPose 래퍼, ICP, antipodal grasp, 빈피킹 파이프라인
├── pdm/                # TimesNet + 비전 멀티모달 PdM
├── common/             # 데이터, 변환, 메트릭(AUROC/PRO), 시각화
└── cli.py              # `mvmm <subcommand>`
configs/                # Hydra: anomaly / data / hardware
scripts/                # download / make_sample_data / smoke_test / train_*/ eval_*
tests/                  # pytest (CPU-only smoke)
docs/                   # ROADMAP, ARCHITECTURE, papers.md
.github/                # CI, PR/Issue 템플릿
Dockerfile · docker-compose.yml · environment.yml · pyproject.toml
```

---

## 🏭 도메인-알고리즘 매핑

| 사용자 작업 | 추천 시작점 | 보강 / 업그레이드 경로 |
|---|---|---|
| 반도체 웨이퍼 패턴 결함 | `anomaly.PatchCore` (WideResNet50) | → `EfficientAD` 실시간 → `AnomalyCLIP` 신규 패턴 zero-shot |
| 자동차 부품 표면 스크래치/덴트 | `anomaly.hybrid.fuse_rule_and_dl` (Canny+LBP + PatchCore) | → ConvNeXt-v2 / DINOv2 백본 swap |
| 치수 검사 (지름/길이/평탄도) | `metrology.SAM2Segmenter` + `circle_fit/line_fit` | → 스테레오(`StereoSGBM`) 또는 Depth Anything v2 |
| 빈피킹 (단일 객체) | `pose.PosePipeline` + `antipodal_grasps` | → FoundationPose + SAM2 (SuperPose) |
| 모터/펌프 예지보전 | `pdm.MultimodalPdMModel` (TimesNet + ResNet) | → PatchTST, TimeLLM 추가 |

---

## 🧪 Development workflow (브랜치 정책)

```
main      ←  안정/배포 가능. 직접 커밋 금지.
└─ dev    ←  통합 브랜치. feature/* PR이 여기로 들어옴.
   └─ feature/<topic>   ←  실제 작업 브랜치
```

```bash
git checkout dev
git pull
git checkout -b feature/anomalyclip-refinement
# work, commit, push
gh pr create --base dev --title "feat(anomaly): refine CLIP prompts"
# after review/CI → merge into dev
# release: dev → main via PR
```

세부 가이드는 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), 로드맵은 [docs/ROADMAP.md](docs/ROADMAP.md), 참고 논문은 [docs/papers.md](docs/papers.md).

---

## 🔬 Status matrix

| Module | Status | Notes |
|---|---|---|
| `anomaly.PatchCore` | ✅ working baseline | greedy coreset + FAISS-optional |
| `anomaly.EfficientAD` | 🟡 architecture only | 학습 루프 `scripts/train_efficient_ad.py` 예정 |
| `anomaly.AnomalyCLIP` | ✅ zero-shot inference | open_clip ViT-B/16 |
| `anomaly.hybrid` | ✅ working | Canny + LBP + Z-score ensemble |
| `metrology.*` | ✅ classical + SAM2/DepthAnything wrappers | wrappers lazy-import |
| `pose.PosePipeline` | ✅ classical seg + antipodal grasps | FoundationPose 옵션 |
| `pdm.MultimodalPdMModel` | ✅ forward pass + CLI 학습 | 데이터 컨버터는 향후 |
| Docker / CI / tests | ✅ pass on CPU runners | smoke + module tests |

---

## License

[MIT](LICENSE) © 2026 CVKim
