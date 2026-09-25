# Laya source verification before E2

Date: 2026-09-25. Status: done; all fixes merged into `src/` before the first
paid E2 run. Log entry: `docs/EXPERIMENTS.md` → "2026-09-25 — pre-E2
verification against Laya's fine-tune notebook".

## Sources read

- `notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb` in
  github.com/NandhaKishorM/laya. This is the run that produced
  `convaiinnovations/laya-typed-decisions`, and it is the training step E2
  ablates.
- `convaiinnovations/laya`: the `model.safetensors` header (206 tensors, read
  with an HTTP range request), `rl_agent_config.json`,
  `typed-decisions/rl_agent_config.json` and `tokenizer/`.
- Earlier reading of `laya/common.py` is in the 2026-09-24 correction entry.

## Findings

### 1. RL advantage normalisation: bug in our port (fixed)

| | Advantage |
|---|---|
| Laya notebook | `adv = r − mean_G(r)`; `adv = adv / (std(adv) + 1e-6)` |
| our port (and the condensed version in `docs/PLAN.md` §2) | `adv = (r − mean_G(r)) / (std(r) + 1e-6)` |

`std(r)` is taken over all noise samples *and* all rows, so it includes the
spread of rewards between rows. It is larger than the within-group spread,
which shrinks the RL term. On a two-row check (one confident row and one
uniform row, σ = 0.5, G = 64), the largest logit-gradient entry was
**0.0036 (old) vs 0.32 (Laya's)**, about 90× weaker. An E2 run on the old code
would have ablated an RL term that was barely there. Regression test:
`tests/test_loss.py::test_rl_advantage_normalised_by_centred_std`.

For the report (§V.3, "variance and advantage normalisation"): after
normalisation the advantages are O(1) whatever the reward scale. The RL
gradient `adv · ε / σ²` then scales like 1/σ, so as σ anneals from 0.4 to 0.1
the RL term's size relative to CE grows by about 4×.

### 2. Spherical weight: the notebook uses 0.75, not the library default 0.5

`laya/common.py::proper_reward` defaults to `w_sph = 0.5`, but the notebook
calls it with `w_sph = 0.75`. The 2026-09-24 correction therefore only holds
for the library default. The E2 configs (`configs/e2/*.yaml`) use 0.75 to
reproduce the notebook. E1's numbers use 0.5 and stand as a study of the
library default.

### 3. Correctness is defined by the gold label

Laya's evaluation scores accuracy and ECE against `gold[qid]["label"]`:

- choice: predicted key vs label
- noul: `p_true ≥ 0.5` vs label
- score: argmax level vs label

It does not use the soft target's argmax. The two disagree on 85 of 8,000
rows in `LocalLLaMA/typed-decisions`: 24 choice, 30 noul and 31 score.
`src/pipelines/eval.py` now uses the label, and also reports soft accuracy
`Σ p·t` and score MAE `|Σ i·p_i − Σ i·t_i|`, as Laya does.

### 4. Laya's weights load into our `DecisionModel`

Tensor names match exactly: `encoder.*` (170 tensors), `head.layers.{0,1}.*`,
`type_emb.weight` and `scorer.{0,1,3}.*`. The two exceptions are
`act_head.*` (Laya's unused escalation head) and a scalar per-type
`temperature` of shape [3]. `TrainingConfig.init_from` drops both. The
tokenizer gives the same ids as `answerdotai/ModernBERT-large`. A CPU check
loaded 201 tensors. On the first 12 test cases (60 question rows), the
*unfine-tuned* `convaiinnovations/laya` got:

| metric | value |
|---|---|
| accuracy (label) | 0.517 |
| mean confidence | 0.682 |
| raw ECE | 0.215 |
| Brier | 0.287 |

It is already over-confident before any typed-decisions fine-tuning.

### 5. Laya's own fitted temperatures are all above 1

The `rl_agent_config.json` of `convaiinnovations/laya` reports per-type
T = **1.64 (choice), 1.25 (score), 1.98 (noul)**. By option bucket: choice:2
1.91, choice:3-5 1.76, noul:2 1.98, score:3-5 1.25, choice:6-10 1.00, and a
degenerate choice:11+ 0.10. T > 1 means the raw outputs are sharper than
calibrated, which matches the §V prediction that the smoothed objective
leaves over-confidence behind. This is supporting evidence only, since
these checkpoints mix RL and CE. The typed-decisions checkpoint's T ≈
1.01–1.06 came from Laya's fixed calibration split and is not comparable.

### 6. Notebook hyperparameters (reproduced in `configs/e2/laya_rlce.yaml`)

| | Laya notebook (2×T4 DDP) | E2 here (1×A100) |
|---|---|---|
| init | `convaiinnovations/laya` | same (`init_from`) |
| epochs | 4 | 4 |
| batch | 8 per GPU × 2 GPUs × accum 4 = 64 | 16 × accum 4 = 64 |
| optimiser | AdamW, wd 0.01; lr 2.5e-5 encoder / 1e-4 head | same (`lr`, `lr_head`) |
| schedule | cosine over total updates, η_min 1e-6 | same (`t_max: auto`) |
| grad clip | 1.0 | 1.0 |
| G, σ | 4; 0.4 → 0.1 linear per epoch | same |
| reward | log + 0.75·spherical − 1.0·RPS (score only) | same |
| CE weight | 1.0 | 1.0 (0.0 in `ce_only`) |
| sequences | max_len 512 / head 192 (items built before the 1024/256 override) | 512 / 192 |
| precision | fp16 autocast + GradScaler | bf16 autocast |
| calibration hold-out | 10% of question rows (≤400), seed 20260922 | 10% of *cases*, seed 42 |
| temperature | one T per type | one T per (type, K-bucket) |

The last three rows are deliberate differences. Case-level holdout keeps
questions from the same case out of both splits.

### 7. Other bugs found while preparing E2

- `configs/train.yaml` had cosine `t_max: 10`. The scheduler steps once per
  optimizer step, so the LR cycled about 34 times per epoch. It is now
  `t_max: auto`.
- `TrainingConfig.amp` was never read. It now drives bf16/fp16 autocast,
  with bf16 on MPS too.
- Only `torch` was seeded. `random` and `numpy` are now seeded as well.

## First E3 observation (tiny encoder, not a result)

A 20-case dry run of the E2 config with `google/bert_uncased_L-2_H-128_A-2`
logged ‖∇RL‖/‖∇CE‖ ≈ 7.5 and cos ≈ 0.70 (w.r.t. the logits) at σ = 0.4. This
is a plumbing check only; the real numbers come from the A100 runs
(`docs/reports/2026-09-25/`).
