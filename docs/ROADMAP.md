# Roadmap

> Pivoted 2026-05-27 to a 4-pillar focus: CCTV tracking + zero-shot
> perception + 3D + PdM/VAD. Each "v0.x" block lands on `dev` as a
> series of `feature/*` PRs.

## v0.2 — Tracking depth
- [ ] Cross-camera Re-ID on Market-1501 with the CLIP backbone
- [ ] DEVA segmentation-tracking integration
- [ ] MASA universal tracker as a tracker option
- [ ] Real CCTV demo video committed (Creative Commons source)
- [ ] BoT-SORT (motion + appearance) tracker

## v0.3 — Zero-shot maturity
- [ ] AnomalyCLIP learnable prompts (Zhou et al. ICLR'24) — fine-tune path
- [ ] GroundingDINO 1.5 (Pro) checkpoint support
- [ ] Grounded-SAM-2 single-shot detect+segment+track entrypoint
- [ ] Florence-2 vision-language wrapper
- [ ] Prompt evaluation harness (precision@k, recall@k for free-form text)

## v0.4 — 3D end-to-end
- [ ] FoundationPose container with prebuilt CUDA extensions
- [ ] MoGe / Marigold depth side-by-side notebook
- [ ] 3DGS end-to-end training notebook (gsplat)
- [ ] Any6D wrapper exercised on a public dataset
- [ ] Stereo-from-CCTV (synthetic pair from monocular + Depth Anything)

## v0.5 — VAD + PdM
- [ ] MemAE training script + ShanghaiTech evaluation
- [ ] MGFN (Multi-Grained Feature Network) — current VAD SOTA
- [ ] PatchTST baseline for PdM (alternative to TimesNet)
- [ ] Cross-attention fusion (sensor ↔ vision) in `pdm.fusion`
- [ ] CMAPSS dataset adapter for RUL regression

## v0.6 — Serving & datasets
- [ ] Full Gradio app with all 4 pillars
- [ ] FastAPI batch inference server with ONNX exports
- [ ] Public-dataset adapters: UCF-Crime, ShanghaiTech, BDD100K-MOT, BOP

## v1.0 — Production hardening
- [ ] ONNX / TensorRT export for tracking + depth + PatchCore
- [ ] Streaming RTSP source + Kafka sink for the tracking pipeline
- [ ] Comprehensive notebook gallery + reproducibility metadata
- [ ] Benchmark suite (latency + accuracy) on a fixed reference clip set
