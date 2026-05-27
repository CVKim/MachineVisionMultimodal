# Experiments — captured results

> Every result on this page was produced by the scripts in `scripts/`
> on the committed demo data + freely-redistributable Ultralytics test
> images (`data/demo/bus.jpg`, `zidane.jpg`). RTX 3080 / CUDA 12.1.

## Reproducing everything

```cmd
:: 1. download tiny demo assets (~200 KB total)
D:\anaconda\envs\anomalydet\python.exe scripts\fetch_demo_assets.py

:: 2. all the runs below
D:\anaconda\envs\anomalydet\python.exe scripts\smoke_test.py
D:\anaconda\envs\anomalydet\python.exe scripts\demo_zero_shot_prompts.py
D:\anaconda\envs\anomalydet\python.exe scripts\eval_prompts.py ^
    --data-root data\sample\widget --prompt-file configs\prompts\anomaly_clip.yaml ^
    --object-name "industrial widget"
D:\anaconda\envs\anomalydet\python.exe scripts\compare_pdm_models.py --epochs 8
D:\anaconda\envs\anomalydet\python.exe scripts\train_memae.py ^
    --train-root data\sample\vad\train\normal --epochs 15 ^
    --output checkpoints\vad_memae.pt
D:\anaconda\envs\anomalydet\python.exe scripts\demo_stereo_from_mono.py ^
    --image data\demo\bus.jpg --out outputs\experiments_v05\stereo

D:\anaconda\envs\anomalydet\Scripts\mvmm.exe track video ^
    --input data\demo\bus_track.mp4 ^
    --output-video outputs\experiments_v05\bus_track.mp4 ^
    --output-json outputs\experiments_v05\bus_track.json ^
    --model yolov8n.pt --classes "person,bus" --frame-rate 15 ^
    --score-threshold 0.30

D:\anaconda\envs\anomalydet\Scripts\mvmm.exe zeroshot detect ^
    --image data\demo\bus.jpg --detector grounding_dino ^
    --classes "person,bus,backpack,handbag" ^
    --output outputs\experiments_v05\bus_gdino.png --score-threshold 0.25

D:\anaconda\envs\anomalydet\Scripts\mvmm.exe depth infer ^
    --image data\demo\bus.jpg ^
    --output outputs\experiments_v05\bus_depth.png ^
    --output-ply outputs\experiments_v05\bus_depth.ply
```

---

## 1. CCTV tracking — YOLOv8n + ByteTrack

Input:  `data/demo/bus_track.mp4` (60 frames @ 15 FPS, synthesized from
        bus.jpg with small per-frame translations).

```
Tracking done: {'n_frames': 60, 'n_detections_total': 174, 'n_unique_track_ids': 6}
  annotated video: outputs\experiments_v05\bus_track.mp4
  per-frame JSON:  outputs\experiments_v05\bus_track.json
```

* 60-frame run finished in **~2.5 s** (~24 FPS effective end-to-end).
* 6 unique track IDs across the clip — matches the 4 people + 1 bus + 1
  background object that YOLOv8n sees on bus.jpg.
* Per-frame results are in the JSON for downstream analytics (zone
  counting, line crossing, dwell timing already supported in
  `mvmm.tracking.analytics`).

---

## 2. Zero-shot detection — GroundingDINO on bus.jpg

```
mvmm zeroshot detect --image data/demo/bus.jpg --detector grounding_dino \
    --classes "person,bus,backpack,handbag" --score-threshold 0.25
```

Captured output (real, unedited):

```
Detected 8 objects -> overlay saved to outputs\experiments_v05\bus_gdino.png
  person        score=0.760  bbox=[ 49.4, 397.6, 246.7, 902.2]
  person        score=0.782  bbox=[222.8, 405.5, 344.8, 858.7]
  person        score=0.847  bbox=[668.2, 394.0, 809.9, 878.4]
  person        score=0.680  bbox=[  0.1, 553.1,  78.4, 872.8]
  bus           score=0.796  bbox=[  0.9, 230.7, 808.4, 744.6]
  handbag       score=0.343  bbox=[223.6, 602.1, 292.4, 707.2]
  handbag       score=0.274  bbox=[223.6, 573.8, 308.1, 707.4]
  backpack      score=0.344  bbox=[223.6, 481.6, 320.5, 708.3]
```

Take-aways:
* 8 detections total — 4 people, 1 bus, 2 handbags, 1 backpack.
* Person scores 0.68–0.85; bus 0.80 — *all from a free-text prompt with
  no fine-tuning on this image*.
* The same prompt could be `"forklift, pallet, hardhat"` for a factory
  feed without any retraining.

---

## 3. Zero-shot AD prompt-engineering effect

`scripts/eval_prompts.py` on `data/sample/widget` (5 test images, MVTec-AD
layout: 2 good + 3 defect), with the four prompt sets in
`configs/prompts/anomaly_clip.yaml`.

| prompt set    | normals | anomalies | gap     | AUROC  | sec |
|---------------|---------|-----------|---------|--------|-----|
| generic       | 0.8792  | 0.8713    | -0.008  | 0.1667 | 11.5 |
| semiconductor | 0.3114  | 0.4283    | +0.117  | **1.0000** | 5.1 |
| automotive    | 0.8775  | 0.8985    | +0.021  | **1.0000** | 5.0 |
| surveillance  | 0.6774  | 0.7726    | +0.095  | **1.0000** | 5.0 |

Take-aways:
* **Generic prompts ("a defective {obj}") are *worse than random* (AUROC
  0.17)** — CLIP cannot disentangle "defective" from "industrial widget"
  without domain anchoring.
* Any of the three *domain-specific* sets achieves perfect ranking
  (AUROC 1.0). Prompt phrasing is the operating-point knob.

This is the same effect AnomalyCLIP / WinCLIP describe in their papers —
captured here on synthetic widget data so you can verify the harness
without a 5 GB MVTec download.

---

## 4. Monocular depth — Depth Anything v2 on bus.jpg

```
mvmm depth infer --image data/demo/bus.jpg \
    --output outputs/experiments_v05/bus_depth.png \
    --output-ply outputs/experiments_v05/bus_depth.ply
```

```
Depth saved outputs\experiments_v05\bus_depth.png  (min=0.251, max=8.250)
PLY saved:  outputs\experiments_v05\bus_depth.ply  (874800 points)
```

* Image is 810x1080 (Ultralytics bus.jpg), depth predicted at full res.
* Depth dynamic range 0.25 — 8.25 (model-relative units; metric scale
  needs a reference patch — pass `--mm-per-px` or use MoGe for absolute).

---

## 5. Stereo-from-monocular synthesis

```
python scripts/demo_stereo_from_mono.py --image data/demo/bus.jpg --out outputs/experiments_v05/stereo
```

```
[stereo] input  : data\demo\bus.jpg  shape=(1080, 810, 3)
[stereo] running Depth Anything v2 ...
[stereo] synthesizing right view ...
[stereo] disparity  min=6.79  max=222.94
```

Take-aways:
* Foreground (people) get ~220 px shift, background ~7 px — exactly the
  right behavior for an 80 mm baseline at 700 px focal.
* Anaglyph + side-by-side written under
  `outputs/experiments_v05/stereo/`.
* Enables stereo-block-matching tests without a real stereo rig.

---

## 6. Image-level anomaly — PatchCore on synthetic widget

`scripts/smoke_test.py` end-to-end run (fit on 10 normals, eval on 5
mixed):

```
[smoke] device = cuda
[smoke] fit done in 2.93s; bank size = torch.Size([1960, 1536])
[smoke] image AUROC = 1.0000
[smoke] PASS
```

* PatchCore memory bank built in **3 s** on RTX 3080.
* 10% coreset of 19,600 patches → 1,960 reference vectors.
* Image-level AUROC 1.0 on the synthetic test split.

---

## 7. PdM model comparison (TimesNet vs PatchTST vs Cross-attn)

`scripts/compare_pdm_models.py --epochs 8` on the committed synthetic
motor data (50 windows of 1024 steps, train/test 70/30 split):

| model                            | accuracy | AUROC  | fit_s | params  |
|----------------------------------|----------|--------|-------|---------|
| TimesNet+ResNet (concat)         | 0.9667   | 0.9956 | 5.7   | 402,946 |
| PatchTST (sensor-only)           | 0.9667   | 0.9956 | 4.5   | 102,594 |
| TimesNet+ResNet (cross-attn)     | 0.9333   | **1.0000** | 4.8 | 519,042 |

Take-aways:
* **PatchTST matches the full multimodal model with 4x fewer
  parameters** — channel-independent transformer is genuinely efficient.
* **Cross-attention** edges out concat on AUROC (1.0 vs 0.996) — the
  ranking is cleaner even when raw accuracy is similar.
* Both transformer variants train in under 5 s on RTX 3080 for this
  toy size; scale up for real-line data.

---

## 8. VAD — frame autoencoder + MemAE on synthetic anomaly clips

Frame autoencoder baseline (`mvmm vad train --epochs 15`):

```
VAD train: samples=160 device=cuda
  epoch 1/15  recon_loss=0.02145
  ...
  epoch 15/15 recon_loss=0.00065
```

Evaluation (`mvmm vad eval`):

```
VAD eval  frames=160  AUROC=0.7990
  test_anomaly_00  mean=0.01706  max=0.04225
  test_anomaly_01  mean=0.01706  max=0.04224
  test_normal_00   mean=0.00067  max=0.00069
  test_normal_01   mean=0.00067  max=0.00069
```

MemAE upgrade (`scripts/train_memae.py --epochs 15`):

```
[memae-train] samples=160 device=cuda n_slots=2000
  epoch 1/15  recon=0.02151  entropy=0.0001
  ...
  epoch 15/15 recon=0.00318  entropy=0.0000
```

* Normal clip score ~ 0.0007, anomaly clip score ~ 0.017 (**~25x ratio**)
  — frames > 10 (after the red intruder enters) drive the gap.
* MemAE's loss curve shows the memory bank converging to a small set of
  active prototypes within a few epochs (entropy → 0 implies
  near-degenerate attention, expected for this toy data; on real
  ShanghaiTech the entropy regularizer keeps the attention soft).

---

## 9. v0.2 BoT-SORT (ByteTrack + appearance gating)

Two unit-tested behaviors (`tests/test_bot_sort.py`):

* `test_botsort_initializes_without_supervision` — bare construction works.
* `test_botsort_remaps_to_lost_track_by_appearance` — when ByteTrack
  emits a new id but the appearance embedding matches a previously seen
  track, BoT-SORT re-maps the id back to the original. This is the core
  occlusion-recovery property the upstream paper claims.

End-to-end CCTV smoke is on the v0.6 roadmap (full benchmark on
MOT17/20 once we add a real-data adapter).

---

## 10. v0.3 Learnable prompts (forward sanity)

```python
from mvmm.zeroshot.learnable_prompt import LearnablePromptCLIP, fit_learnable_prompts

model = LearnablePromptCLIP(classnames=["normal industrial widget",
                                        "defective industrial widget"])
losses = fit_learnable_prompts(model, images=widget_few_shot_imgs,
                                labels=widget_few_shot_labels,
                                epochs=20, lr=5e-3)
print(losses[0], losses[-1])  # decreasing
```

* Only `model.ctx` (12 × 512 = 6,144 parameters) is trainable.
* CLIP image/text encoders are frozen, so the AnomalyCLIP-style few-shot
  fine-tune lands in a budget you can run on a laptop CPU.

---

## What is *not* on this page

These items require a separate external install + larger downloads,
so they're scaffolded as wrappers but not benchmarked here:

* FoundationPose, Any6D — need a CUDA-compiled native build.
* 3D Gaussian Splatting (`gsplat`) training — needs a custom CUDA wheel.
* Marigold / MoGe depth — needs `diffusers` + the MoGe pip package.
* Florence-2 — wrapper is committed; large 770 MB checkpoint downloads
  the first time you call `Florence2()`.
* UCF-Crime / ShanghaiTech full VAD training — adapters land in v0.5.

For each, the README's Section 16 (Troubleshooting) explains the
install path; once installed, the same CLI commands work.
