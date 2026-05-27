# SOTA reference list (2022–2026)

Curation criteria: applicability to industrial CCTV / 3D / PdM,
public code, current influence as of 2026.

## CCTV / video tracking

| Year | Paper | Module | Note |
|---|---|---|---|
| 2022 | ByteTrack — Zhang et al. ECCV | `tracking.byte_track` | Standard online MOT baseline |
| 2022 | BoT-SORT — Aharon et al. | (planned) | Motion + appearance |
| 2024 | SAM2 — Ravi et al. | `tracking.sam2_video` | Promptable video segmentation |
| 2024 | DEVA — Cheng et al. | (planned) | Decoupled video segmentation tracking |
| 2024 | MASA — Li et al. | (planned) | Match Anything by Segmenting Anything |
| 2024 | YOLO-World — Cheng et al. | `tracking.detectors` (via ultralytics) | Open-vocab YOLO |
| 2024 | RT-DETR — Lv et al. | `tracking.detectors` (via ultralytics) | Real-time DETR |
| 2024 | CoTracker3 — Meta | (planned) | Long-range point tracking |
| 2024 | OSNet / TransReID | `tracking.reid` | Cross-camera appearance |

## Zero-shot / open-vocabulary

| Year | Paper | Module | Note |
|---|---|---|---|
| 2023 | Grounding DINO — Liu et al. | `tracking.detectors`, `zeroshot.grounding_dino` | Text → boxes |
| 2023 | OWLv2 — Minderer et al. | `zeroshot.owl_v2` | HF transformers wrapper |
| 2023 | WinCLIP — Jeong et al. CVPR | `zeroshot.anomaly_clip` style | Zero-shot AD via CLIP windows |
| 2024 | SAM2 — Ravi et al. | `zeroshot.sam2_promptable` | Promptable segmentation |
| 2024 | AnomalyCLIP — Zhou et al. ICLR | (planned learnable prompts) | Object-agnostic AD prompts |
| 2024 | Florence-2 — Microsoft | (planned) | Unified VL foundation |
| 2024 | Grounded-SAM-2 | `zeroshot.pipeline` style | Detect + segment in one go |
| 2025 | MultiADS | (planned) | Defect-aware zero-shot AD |
| 2025 | MuSc-V2 — arXiv 2511.10047 | (planned) | Training-free multimodal AD (2D+3D) |
| 2025 | FP-CLIP | (planned) | Low-FPR zero-shot AD |

## 3D

| Year | Paper | Module | Note |
|---|---|---|---|
| 2023 | 3D Gaussian Splatting — Kerbl et al. SIGGRAPH | `three_d.gaussian_splatting` (gsplat backend) | Real-time radiance fields |
| 2024 | FoundationPose — Wen et al. CVPR (highlight) | `three_d.pose.foundation_pose` | Unified 6D pose + tracking |
| 2024 | Depth Anything v2 — Yang et al. | `three_d.depth.DepthAnythingV2` | Best-in-class mono depth |
| 2024 | Marigold — Ke et al. CVPR | `three_d.depth.monocular_extras.Marigold` | Diffusion-prior depth |
| 2024 | SuperPose — Yan et al. | (planned) | FoundationPose + SAM2 + LightGlue |
| 2025 | Any6D — Liu et al. CVPR | `three_d.pose.any6d` | Model-free 6D pose |
| 2025 | MoGe — Wang et al. CVPR | `three_d.depth.monocular_extras.MoGe` | Geometric foundation model |
| 2024 | AnyGrasp — Fang et al. | (planned) | Universal grasp detection |

## Predictive maintenance / time-series

| Year | Paper | Module | Note |
|---|---|---|---|
| 2022 | PatchTST — Nie et al. | (planned) | Channel-independent transformer |
| 2023 | TimesNet — Wu et al. ICLR | `pdm.timeseries` | 2D periodic reshape |
| 2024 | TimeLLM — Jin et al. | (planned) | LLM-aligned forecasting |
| 2024 | GPT4TS — Zhou et al. | (planned) | Frozen LLM time-series |

## Video anomaly detection

| Year | Paper | Module | Note |
|---|---|---|---|
| 2019 | MemAE — Gong et al. ICCV | `vad.memae` | Memory-augmented AE baseline |
| 2022 | MGFN — Chen et al. AAAI | (planned) | Multi-grained feature network |
| 2024 | MULDE — Micorek et al. CVPR | (planned) | Multi-scale latent denoising |
| 2024 | AnomalyRuler | (planned) | LLM-augmented VAD reasoning |

## Image-level anomaly detection (kept from v0.1)

| Year | Paper | Module | Note |
|---|---|---|---|
| 2022 | PatchCore — Roth et al. CVPR | `anomaly.patchcore` | Memory-bank baseline |
| 2024 | EfficientAD — Batzner et al. WACV | `anomaly.efficient_ad` | Real-time S/T + AE |

## Surveys

- [Awesome industrial anomaly detection](https://github.com/M-3LAB/awesome-industrial-anomaly-detection)
- [Awesome object pose estimation (IJCV'26 survey)](https://github.com/CNJianLiu/Awesome-Object-Pose-Estimation)
