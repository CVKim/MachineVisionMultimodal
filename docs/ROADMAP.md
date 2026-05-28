# Roadmap

> Pivoted 2026-05-27 to a 4-pillar focus (CCTV tracking + zero-shot +
> 3D + PdM/VAD). v0.2 / v0.3 / v0.4 / v0.5 are now substantially
> landed — see [docs/EXPERIMENTS.md](EXPERIMENTS.md) for captured
> results.

Legend: ✅ done · 🟡 wrapper-only (needs external install) · ⬜ planned

## v0.2 — Tracking depth
- ✅ BoT-SORT (ByteTrack + appearance gating) — `mvmm.tracking.bot_sort`
- ✅ CLIP-based appearance ReID — `mvmm.tracking.reid.CLIPReID`
- ✅ OSNet ReID wrapper — `mvmm.tracking.reid.OSNetReID`
- ✅ Real CCTV demo clips fetched + tracked end-to-end on 4 videos
      (`scripts/demo_cctv_videos.py`, see [EXPERIMENTS.md §11](EXPERIMENTS.md#11-real-public-cctv--yolov8s--bytetrack-on-4-clips))
- ✅ Line-counter analytics demo on people_detection.mp4 (1 in / 7 out)
- ✅ Pose-based activity recognition (idle / walking / working / lifting)
      with per-track time accounting — `mvmm.tracking.pose_activity` +
      `scripts/run_activity_recognition.py`, see [EXPERIMENTS.md §11b](EXPERIMENTS.md)
- ⬜ DEVA segmentation tracking (needs DEVA install)
- ⬜ MASA universal tracker (needs MASA install)
- ⬜ Market-1501 full ReID benchmark (needs dataset download)

## v0.3 — Zero-shot maturity
- ✅ AnomalyCLIP learnable prompts — `mvmm.zeroshot.learnable_prompt`
- ✅ Few-shot training loop — `fit_learnable_prompts`
- ✅ `LearnableAnomalyCLIP` drop-in detector
- ✅ Prompt evaluation harness — `scripts/eval_prompts.py`
- ✅ Prompt configuration registry — `configs/prompts/anomaly_clip.yaml`
- ✅ GroundingDINO 1.5 API compatibility (threshold/box_threshold)
- ✅ Grounded-SAM-2 single-shot detect+segment pipeline (`zeroshot.OpenVocabPipeline`)
- 🟡 Florence-2 wrapper — `mvmm.zeroshot.florence2` (770 MB on first call)

## v0.4 — 3D end-to-end
- ✅ Depth comparison script (Depth Anything v2 / MoGe / Marigold) — `scripts/compare_depth.py`
- ✅ Stereo-from-monocular synthesis — `mvmm.three_d.depth.stereo_synth`
- ✅ Stereo synth demo — `scripts/demo_stereo_from_mono.py`
- 🟡 FoundationPose wrapper — needs upstream install (set MVMM_FOUNDATIONPOSE_PATH)
- 🟡 Any6D wrapper — needs upstream install (set MVMM_ANY6D_PATH)
- 🟡 3DGS training — needs CUDA-compiled gsplat
- 🟡 MoGe / Marigold wrappers — need `diffusers` / `moge`

## v0.5 — VAD + PdM depth
- ✅ MemAE (memory-augmented AE) — `mvmm.vad.memae`
- ✅ MemAE training script with entropy regularizer — `scripts/train_memae.py`
- ✅ PatchTST baseline — `mvmm.pdm.patchtst`
- ✅ Cross-attention multimodal fusion — `mvmm.pdm.fusion_attn`
- ✅ 3-way model comparison harness — `scripts/compare_pdm_models.py`
- ⬜ MGFN VAD baseline (Chen et al. AAAI'22)
- ⬜ MULDE VAD baseline (Micorek et al. CVPR'24)
- ⬜ UCF-Crime / ShanghaiTech dataset adapters
- ⬜ CMAPSS / N-CMAPSS RUL regression adapter

## v0.6 — Serving & datasets
- ✅ Gradio 3-tab app — `mvmm.serving.gradio_app`
- ⬜ FastAPI batch inference server
- ⬜ ONNX export for tracking + depth + PatchCore
- ⬜ Public-dataset adapters: BDD100K-MOT, BOP, MOT17/20

## v1.0 — Production hardening
- ⬜ TensorRT optimization for tracking + depth
- ⬜ Streaming RTSP source + Kafka sink for tracking pipeline
- ⬜ Comprehensive notebook gallery
- ⬜ Benchmark suite (latency + accuracy) on a fixed reference clip set
