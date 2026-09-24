# scripts/

Utility entrypoints. Run them from the repo root, e.g.
`uv run python scripts/<name>.py ...`.

```
scripts/
└── training/   train / evaluate / smoke-test
```

Data comes directly from the Hugging Face Hub (`LocalLLaMA/typed-decisions`,
see `src.pipelines.config.TrainingConfig.dataset_name`) rather than a local
generator, so there is no `scripts/data/` step to run first.

## training/

| Script | Purpose |
|---|---|
| `training/train.py` | Thin wrapper over `src.pipelines.train` (the real CLI) |
| `training/evaluate.py` | Evaluate a checkpoint's raw & post-temperature calibration on the test split, optional JSON summary |
| `training/smoke_test.py` | End-to-end run against the real dataset with a tiny encoder (`configs/rlcd_smoke.yaml`): train, evaluate, assert finite metrics |
