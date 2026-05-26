# SOTA reference list (2022–2026)

> 큐레이션 기준: 제조 현장 적용 가능성 + 코드 공개 + 2026년 현재 영향력.

## Anomaly / defect detection

| Year | Paper | Module | Note |
|---|---|---|---|
| 2022 | PatchCore — Roth et al. CVPR | `anomaly.patchcore` | Memory-bank baseline. WideResNet50 / DINOv2 backbones. |
| 2023 | WinCLIP — Jeong et al. CVPR | inspires `anomaly.anomaly_clip` | Zero-shot multi-scale window scoring. |
| 2023 | DRAEM — Zavrtanik et al. | (planned) | Reconstruction-based, strong on logical defects. |
| 2024 | EfficientAD — Batzner et al. WACV | `anomaly.efficient_ad` | PDN S/T + AE, ms-level inference. |
| 2024 | AnomalyCLIP — Zhou et al. ICLR | `anomaly.anomaly_clip` (next) | Object-agnostic learnable prompts. |
| 2024 | M3DM — Wang et al. | (planned) | RGB + point-cloud multimodal AD. |
| 2025 | FP-CLIP | (planned) | Low-FPR zero-shot extension. |
| 2025 | MultiADS — Tilli et al. arXiv:2504.06740 | (planned) | Defect-aware supervision multi-type. |
| 2025 | DMA + SAP (CLIP-DINOv2 fusion) — MDPI Electronics 14/24 | (planned) | 93.4% image-AUROC across 7 datasets. |
| 2025 | MuSc-V2 — arXiv:2511.10047 | (planned) | Training-free multimodal (2D + 3D). |

## Dimensional metrology / segmentation / depth

| Year | Paper | Module | Note |
|---|---|---|---|
| 2024 | SAM2 — Ravi et al. | `metrology.segmentation.SAM2Segmenter` | Promptable segmentation, video memory. |
| 2024 | Depth Anything v2 — Yang et al. | `metrology.depth.DepthAnythingV2` | Best-in-class monocular depth. |
| 2025 | Grounded-SAM-2 | (planned) | Text→mask for inspection points. |

## 6D pose / bin picking

| Year | Paper | Module | Note |
|---|---|---|---|
| 2022 | MegaPose — Labbé et al. | (planned wrapper) | Multi-view refinement with CAD. |
| 2024 | FoundationPose — Wen et al. CVPR (highlight) | `pose.foundation_pose` | Unified model + tracking. |
| 2024 | SuperPose — Yan et al. arXiv:2409.19986 | (planned) | FoundationPose + SAM2 + LightGlue. |
| 2025 | Any6D — CVPR | (planned) | Model-free, no CAD. |
| 2025 | XYZ-ibd | (dataset) | Industrial bin-picking benchmark. |
| 2024 | AnyGrasp — Fang et al. | (planned) | Universal grasp detection. |

## Predictive maintenance / multimodal time series

| Year | Paper | Module | Note |
|---|---|---|---|
| 2022 | PatchTST — Nie et al. | (planned) | Channel-independent transformer. |
| 2023 | TimesNet — Wu et al. ICLR | `pdm.timeseries` | 2D periodic reshape. |
| 2024 | TimeLLM — Jin et al. | (planned) | LLM-aligned forecasting. |
| 2024 | GPT4TS — Zhou et al. | (planned) | Frozen LLM time-series. |

## Generic surveys (recommended reading)

- "Awesome industrial anomaly detection" (M-3LAB) — paper list and datasets.
- "Awesome object pose estimation" (CNJianLiu) — IJCV 2026 survey.
