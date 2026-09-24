# Reverse-Engineering RLCD

A technical report project analysing what "Reinforcement Learning for
Calibrated Decisions" (RLCD) does — and doesn't do — for typed probabilistic
decisions, using TypeSafe's closed **Jev** model (public claims only) and its
open reproduction attempt **[Laya](https://github.com/NandhaKishorM/laya)**
(architecture + training loop verified from source) as the two reference
points. Output is an IEEE-format report (6–8 pages, arXiv + GitHub) plus the
small, reproducible experiments that back its central claim: Laya's RL term
is a score-function (evolution-strategies) estimator of a noise-smoothed
proper scoring rule, and that smoothing biases the noise-free inference-time
distribution toward over-confidence as σ grows.

Start here:

- **[`docs/PLAN.md`](docs/PLAN.md)** — the full project brief: background on
  Jev/Laya, Laya's verified internals, the central analytical finding, the
  experiment matrix (E1–E5), datasets, and the IEEE report outline.
- **[`docs/RESEARCH.md`](docs/RESEARCH.md)** — the distilled research
  question, scope, and method derived from the plan.
- **[`docs/TODO.md`](docs/TODO.md)** — the step-by-step task list from this
  scaffold to a submitted report.
- **[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)** — the runbook/log for each
  gated experiment (hypothesis, config, command, result, decision).
- **[`docs/analysis/`](docs/analysis/)** — standalone analytical notes (e.g.
  the noise-smoothing over-confidence proof sketch) that feed the report.
- **[`experiments/`](experiments/)** — self-contained numerical scripts (E1's
  toy bias check) that don't need the full `src/` training pipeline.

## Scaffold status

This repository started from a generic MLP-over-fixed-features research
template (config-driven training loop, callbacks, evaluation, tests). Its own
"Extending the template" contract — replace the model/dataset/loss/metrics,
keep the checkpoint schema, CLI, and callbacks working — is exactly what has
been done: `src/` now implements the typed-decision (RLCD/Laya-style) model
needed for E2 (docs/PLAN.md sec. 5's loss ablation) directly in place, not as
a parallel package. See [Repository layout](#repository-layout) below for
what each file holds now.

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
git clone <your-repo-url> rlcd-reverse-engineering
cd rlcd-reverse-engineering

uv sync                # core: torch, transformers, datasets, numpy, scikit-learn
uv sync --extra rich   # optional: pretty training summary (panels / tables)
uv sync --extra wandb  # optional: Weights & Biases logging
uv run pytest          # test suite (offline — uses fake encoder/tokenizer fixtures)

# end-to-end against the real dataset with a tiny real encoder, no GPU required
uv run python scripts/training/smoke_test.py

# E1's standalone toy-bias experiment (no src/ pipeline needed)
uv run python experiments/e1_toy_bias.py
```

Platform-aware torch builds resolve from `pyproject.toml`: Linux + NVIDIA uses
the CUDA wheels, macOS resolves CPU/MPS wheels.

## Data format

Training reads directly from a Hugging Face dataset shaped like
`LocalLLaMA/typed-decisions` (docs/PLAN.md sec. 6): each row is one *case*
with three JSON-string columns —

| Column | Contents |
|---|---|
| `state` | free-text/JSON context the questions are asked about |
| `questions` | `{qid: {"type": "choice"\|"score"\|"noul", "instructions": str, "criteria": ...}}` |
| `gold` | `{qid: {"label": str, "probabilities": {option_key: float, ...}, ...}}` |

`src.data.TypedDecisionDataset` flattens each case into one row per question
(state re-encoded per question, matching Laya's own sequence builder) and
`src.data.build_sequence` renders that row as
`[CLS] <type> question: <instr> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] <state> [SEP]`.
A calibration slice is carved out of the train split at load time
(`calib_fraction` in `TrainingConfig`) for raw-ECE validation and
post-training temperature fitting — never from the test split
(docs/PLAN.md sec. 5, Laya issue #186).

## Training

```bash
uv run python scripts/training/train.py --config configs/train.yaml

# individual overrides win over the YAML
uv run python scripts/training/train.py \
    --config configs/train.yaml --epochs 5 --lr 1e-5 --w_rl 0 --w_ce 1

# fast local smoke run: tiny real encoder, small batch, one epoch
uv run python scripts/training/train.py --config configs/rlcd_smoke.yaml
```

The run writes `checkpoints/<run_name>/best.pt`, periodic `epoch_*.pt`, and a
`train_log.json`. Checkpoints carry the model's `DecisionModelConfig` so the
evaluator/inference script can rebuild it without the original YAML. `w_rl`
/ `w_ce` / `sigma_start` / `sigma_end` / `anneal_sigma` in `TrainingConfig`
are exactly the E2 ablation knobs from docs/PLAN.md sec. 5 (CE-only:
`w_rl: 0`; RL-only: `w_ce: 0`; fixed-σ: `anneal_sigma: false`).

## Evaluation and inference

```bash
# fit temperature on a calibration slice, report raw + post-T metrics on test
uv run python scripts/training/evaluate.py \
    --ckpt checkpoints/<run>/best.pt --json results/eval.json

# score one typed question against one state blob
uv run python -m src.pipelines.infer \
    --ckpt checkpoints/<run>/best.pt \
    --state '{"task": "..."}' \
    --question '{"type": "choice", "instructions": "...", "criteria": {"a": "...", "b": "..."}}'
```

## Repository layout

```
src/
├── config.py            # DecisionModelConfig — model architecture only; QTYPES
├── data.py              # TypedDecisionDataset, build_sequence, collate_fn, calib split
├── modules/
│   ├── model.py         # DecisionModel: encoder + transformer head + marker readout
│   └── loss.py          # rlcd_loss (RL+CE), proper_reward, probability/label helpers
├── pipelines/
│   ├── config.py        # TrainingConfig — training-loop hyperparameters (incl. RLCD loss weights)
│   ├── train.py         # training loop + callback wiring
│   ├── eval.py          # raw/post-temperature ECE, Brier, NLL, temperature fitting
│   └── infer.py         # single-checkpoint inference
├── callbacks/           # checkpoint, early_stopping, lr_scheduler, wandb — unmodified, domain-agnostic
└── utils/               # io, model, device helpers — unmodified
configs/                 # training YAML configs (train.yaml, rlcd_smoke.yaml)
experiments/             # standalone numerical scripts (E1 toy bias)
scripts/
└── training/            # train, evaluate, smoke_test
tests/                   # pytest suite (conftest.py has offline fake tokenizer/encoder fixtures)
docs/                    # research plan, notes, experiment log, analysis notes
notebooks/               # exploratory notebooks
```

## Configuration

Configuration is split in two, mirroring the reference project:

- **`src/config.py` → `DecisionModelConfig`** holds the *architecture*
  (encoder name, head depth, sequence budgets, K-bucket boundaries). It is
  saved into every checkpoint.
- **`src/pipelines/config.py` → `TrainingConfig`** holds the *training loop*
  (dataset, optimizer, RLCD loss weights, callbacks). It is loaded from
  `configs/train.yaml` and overridable from the CLI.

Callbacks are wired from the same YAML, unchanged from the inherited
scaffold:

```yaml
lr_scheduler: {type: cosine, t_max: 10, eta_min: 1.0e-7}
early_stopping: {enabled: true, monitor: raw_ece, mode: min, patience: 3}
save_best: true
best_metric: raw_ece
best_mode: min
wandb: {enabled: false, project: rlcd-reverse-engineering}
```

## Extending the template

1. Replace `DecisionModel` in `src/modules/model.py` (keep the
   `forward(ids, attention_mask, marker_pos, marker_mask, qtype) -> logits`
   contract).
2. Replace `TypedDecisionDataset` / `build_sequence` in `src/data.py` for a
   different typed-decision source.
3. Replace the metrics in `src/pipelines/eval.py`.
4. Wire any new callback into `src/callbacks/` and register it in
   `build_callbacks` (`src/pipelines/train.py`).

Everything else — checkpoint schema, CLI, W&B/early-stopping/best-checkpoint
callbacks, test structure — keeps working.

## License

Released under the repository [LICENSE](LICENSE).
