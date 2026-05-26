# Architecture

## High-level layering

```
                ┌────────────────────────────────────────┐
                │   CLI (`mvmm` Typer)                   │
                │   Hydra configs (configs/)             │
                └──────────────┬─────────────────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
┌───────▼───────┐    ┌─────────▼────────┐    ┌────────▼────────┐
│  anomaly      │    │   metrology      │    │   pose          │
│  base/        │    │   calibration    │    │   icp_classical │
│  patchcore    │    │   segmentation   │    │   foundation_pose
│  efficient_ad │    │   depth          │    │   grasp         │
│  anomaly_clip │    │   measure        │    │   pipeline      │
│  hybrid       │    └──────────────────┘    └─────────────────┘
└───────────────┘
                                ┌──────────┐
                                │  pdm     │
                                │  timesNet│
                                │  vision  │
                                │  fusion  │
                                │  datasets│
                                └──────────┘
                       ▲
       ┌───────────────┴───────────────┐
       │     mvmm.common               │
       │  data / transforms / metrics  │
       │  io / viz                     │
       └───────────────────────────────┘
                       ▲
                       │
              torch / opencv / open_clip / open3d / faiss
```

## Design principles

1. **Common base, pluggable backends.** Every anomaly detector subclasses
   `AnomalyDetector`; every segmentation backend exposes `__call__(image_rgb, points|box)`.
   Backends import their heavy deps *lazily* so unrelated CLI paths stay snappy.

2. **CPU-importable, GPU-runnable.** `mvmm` must import on a fresh CPU
   machine without CUDA. GPU-only modules (e.g. `AnomalyCLIP`) gate on
   `torch.cuda.is_available()` and fall back to CPU silently.

3. **Classical ⇔ DL fusion is first-class.** `mvmm.anomaly.hybrid` is not
   an afterthought — manufacturing tolerances often need a deterministic
   floor under the DL stack.

4. **One source of truth per dataset.** `MVTecADDataset` is reused across
   anomaly modules; metrology samples wrap it the same way.

5. **Configs over flags.** Hydra (`configs/`) drives reproducible runs;
   the CLI exposes the same knobs for one-off use.

## Folder roles

| Folder | Purpose |
|---|---|
| `src/mvmm/` | Importable package — every public name comes from here |
| `configs/` | Hydra YAML configs (anomaly/data/hardware) |
| `scripts/` | Top-level entrypoints (downloads, train/eval, smoke) |
| `tests/` | CPU-only smoke + interface tests (CI gate) |
| `docs/` | ROADMAP, ARCHITECTURE, papers.md |
| `data/` | Gitignored datasets (`data/sample/` is committed for CI) |
| `checkpoints/` | Gitignored saved models |
| `outputs/` | Gitignored predictions / heatmaps |

## How to add a new SOTA method

1. **Pick the right module folder** (`anomaly/`, `metrology/`, `pose/`, `pdm/`).
2. **Subclass the existing base** (`AnomalyDetector`, etc.) or follow the
   `__call__(...)`-style backend interface.
3. **Lazy-import** any heavy deps inside `__init__` or method bodies.
4. **Add a config YAML** under `configs/<module>/<your_method>.yaml`.
5. **Wire a CLI subcommand** if it deserves a top-level UX, else expose via Hydra.
6. **Add a smoke test** in `tests/test_<module>.py`.
7. **Update `docs/papers.md`** and the README status matrix.
