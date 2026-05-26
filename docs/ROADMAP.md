# Roadmap

> 활성 작업은 `dev` 브랜치에서 feature/* 로 진행. 완료된 항목은 체크.

## v0.2 — Anomaly detection deep-dive
- [ ] EfficientAD full training loop with hard-feature loss (`scripts/train_efficient_ad.py`)
- [ ] DINOv2 backbone option for PatchCore (anomaly-detection SOTA shift in 2024–2026)
- [ ] AnomalyCLIP: learnable object-agnostic prompts (Zhou et al., ICLR'24)
- [ ] MultiADS (defect-aware supervision) integration
- [ ] Visualization dashboard via Gradio

## v0.3 — Metrology
- [ ] SAM2 video predictor for moving parts on a conveyor
- [ ] Depth Anything v2 metric-mode + uncertainty estimate
- [ ] Stereo + structured-light helpers (calibrated rig)
- [ ] Tutorial notebook: pitch/diameter measurement on a real part

## v0.4 — 6D pose & bin picking
- [ ] FoundationPose integration via container (third_party/FoundationPose submodule)
- [ ] Any6D model-free pipeline (CVPR'25) — no CAD path
- [ ] AnyGrasp / GraspNet weights wrapper
- [ ] XYZ-ibd benchmark adapter

## v0.5 — Predictive maintenance
- [ ] PatchTST baseline alongside TimesNet
- [ ] Cross-attention fusion (instead of late-fusion concat)
- [ ] CMAPSS / N-CMAPSS dataset adapters for RUL regression
- [ ] LLM-augmented log analysis (TimeLLM-style)

## v1.0 — Production-ready
- [ ] ONNX / TensorRT export for PatchCore + EfficientAD
- [ ] FastAPI inference server (`uvicorn mvmm.serve:app`)
- [ ] Streamlit demo with file upload + heatmap
- [ ] Comprehensive examples notebook gallery

## Tracking
이슈는 GitHub Projects 보드에서 관리합니다 (`.github/ISSUE_TEMPLATE/feature_request.md`).
