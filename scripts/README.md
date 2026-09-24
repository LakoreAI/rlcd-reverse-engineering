# scripts/

Utility entrypoints, grouped by category. Run them from the repo root, e.g.
`uv run python scripts/<category>/<name>.py ...`.

```
scripts/
├── data/       dataset preparation / synthetic generators
└── training/   train / evaluate / smoke-test
```

## data/

| Script | Purpose |
|---|---|
| `data/make_synthetic.py` | Generate a train/val/test `.npz` feature dataset for smoke runs |

## training/

| Script | Purpose |
|---|---|
| `training/train.py` | Thin wrapper over `src.pipelines.train` (the real CLI) |
| `training/evaluate.py` | Evaluate a checkpoint on a feature split, optional JSON summary |
| `training/smoke_test.py` | End-to-end synthetic run: train, evaluate, assert finite metrics |
