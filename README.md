# AI Research Project Template

A small, opinionated scaffold for ML research code. It pairs a **config-driven
training loop** with reusable **callbacks**, a standalone **evaluation** path,
and a **test suite** — so an experiment is reproducible from a YAML file plus a
checkpoint.

The reference model is deliberately trivial: a multilayer perceptron (MLP) with
a linear classification head over fixed-size feature vectors. Swap the model,
dataset, and metrics for your own task; the surrounding structure (config
splits, pipelines, callbacks, scripts, tests) is what the template provides.

## Contents

- [Quickstart](#quickstart)
- [Data format](#data-format)
- [Training](#training)
- [Evaluation and inference](#evaluation-and-inference)
- [Repository layout](#repository-layout)
- [Configuration](#configuration)
- [Extending the template](#extending-the-template)
- [License](#license)

## Quickstart

Requires Python ≥ 3.12 and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone <your-repo-url> ai-research-project-template
cd ai-research-project-template

uv sync                # core: torch, numpy, scikit-learn
uv sync --extra rich   # optional: pretty training summary (panels / tables)
uv sync --extra wandb  # optional: Weights & Biases logging
uv run pytest          # test suite

# end-to-end on synthetic features, no dataset required
uv run python scripts/training/smoke_test.py
```

Platform-aware torch builds resolve from `pyproject.toml`: Linux + NVIDIA uses
the CUDA wheels, macOS resolves CPU/MPS wheels.

## Data format

The template trains on fixed-size feature matrices stored as `.npz` files with
two arrays:

| Array | Shape | Dtype | Meaning |
|---|---|---|---|
| `X` | `(N, input_dim)` | `float32` | one feature vector per example |
| `y` | `(N,)` | `int64` | class label in `[0, num_classes)` |

Expected layout:

```
data/
└── raw/
    ├── train.npz    # required
    ├── val.npz      # optional (held-out split for val_loss / early stopping)
    └── test.npz     # optional (reported metrics)
```

Generate a synthetic dataset to try the pipeline:

```bash
uv run python scripts/data/make_synthetic.py \
    --out data/raw --n_samples 4000 --n_features 32 --n_classes 5
```

## Training

```bash
uv run python scripts/training/train.py --config configs/train.yaml

# individual overrides win over the YAML
uv run python scripts/training/train.py \
    --config configs/train.yaml --epochs 50 --lr 5e-4 --batch_size 64
```

The run writes `checkpoints/<run_name>/best.pt`, periodic `epoch_*.pt`, and a
`train_log.json`. Checkpoints carry the model config and label mapping so the
evaluator can rebuild the model without the original YAML.

## Evaluation and inference

```bash
# evaluate a checkpoint on a split
uv run python scripts/training/evaluate.py \
    --ckpt checkpoints/<run>/best.pt --data data/raw/test.npz

# score a single feature file
uv run python -m src.pipelines.infer \
    --ckpt checkpoints/<run>/best.pt --features data/raw/test.npz --index 0
```

## Repository layout

```
src/
├── config.py            # MLPConfig — model architecture only
├── data.py              # FeatureDataset, npz loading, synthetic generator
├── modules/
│   ├── model.py         # MLPBackbone + MLPClassifier (logits, feature)
│   └── loss.py          # ClassificationLoss + probability/label helpers
├── pipelines/
│   ├── config.py        # TrainingConfig — training-loop hyperparameters
│   ├── train.py         # training loop + callback wiring
│   ├── eval.py          # accuracy / macro-F1 / per-class report
│   └── infer.py         # single-checkpoint inference
├── callbacks/           # checkpoint, early_stopping, lr_scheduler, wandb
└── utils/               # io, model, device helpers
configs/                 # training YAML configs
scripts/
├── data/                # synthetic dataset generator
└── training/            # train, evaluate, smoke_test
tests/                   # pytest suite
docs/                    # research notes, experiment log
notebooks/               # exploratory notebooks
```

## Configuration

Configuration is split in two, mirroring the reference project:

- **`src/config.py` → `MLPConfig`** holds the *architecture* (input/output
  dims, hidden layers, dropout, ...). It is saved into every checkpoint.
- **`src/pipelines/config.py` → `TrainingConfig`** holds the *training loop*
  (data paths, optimizer, callbacks). It is loaded from `configs/train.yaml`
  and overridable from the CLI.

Callbacks are wired from the same YAML:

```yaml
lr_scheduler: {type: cosine, t_max: 100, eta_min: 1.0e-5}
early_stopping: {enabled: true, monitor: accuracy, mode: max, patience: 10}
save_best: true
best_metric: accuracy
best_mode: max
wandb: {enabled: false, project: ai-research-template}
```

## Extending the template

1. Replace `MLPClassifier` in `src/modules/model.py` (keep the
   `forward(x) -> (logits, feature)` contract).
2. Replace `FeatureDataset` in `src/data.py` for your input format.
3. Replace the metrics in `src/pipelines/eval.py`.
4. Wire any new callback into `src/callbacks/` and register it in
   `build_callbacks` (`src/pipelines/train.py`).

Everything else — checkpoint schema, CLI, W&B/early-stopping/best-checkpoint
callbacks, tests — keeps working.

## License

Released under the repository [LICENSE](LICENSE).
