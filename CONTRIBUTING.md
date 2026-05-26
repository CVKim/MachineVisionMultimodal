# Contributing

## Branch policy
- `main` is protected. Direct pushes disallowed.
- `dev` is the integration branch.
- Work happens on `feature/<short-topic>` branches off `dev`.
- PR target: `dev`. Releases merge `dev → main` via PR.

## Local setup
```bash
conda env create -f environment.yml
conda activate mvmm
pre-commit install
```

## Before opening a PR
```bash
ruff check src tests scripts
ruff format src tests scripts
pytest -m "not slow and not gpu and not network"
python scripts/smoke_test.py
```

## Commit style
Conventional Commits encouraged:
- `feat(anomaly): add DINOv2 backbone to PatchCore`
- `fix(metrology): handle empty mask in dimension_from_mask`
- `docs(readme): clarify zero-shot quickstart`
- `refactor(pdm): switch fusion to cross-attention`

## Code style
- Ruff config in `pyproject.toml` (line length 110).
- Type hints expected on all public functions.
- Lazy-import heavy backends inside functions.
