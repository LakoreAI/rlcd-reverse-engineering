# Experiments

Runbook for gated experiments. Each experiment records: hypothesis, config,
command, result, and decision.

## Template

### <YYYY-MM-DD> — <short name>

- **Hypothesis:**
- **Config:** `configs/<file>.yaml`
- **Command:**

  ```bash
  uv run python scripts/training/train.py --config configs/<file>.yaml
  ```

- **Result:** _accuracy / macro-F1 / loss, plus where the log lives._
- **Decision:** _adopt / discard / follow up with ...

## Log

### 2026-09-24 — E1: toy smoothing bias

- **Hypothesis:** Minimising the noise-smoothed log score
  `E_eps[-t·log softmax(z+eps)]`, `eps ~ N(0, σ²I)` zero-mean projected,
  yields a noise-free `softmax(z*)` that is systematically sharper than the
  target `t`, growing with `σ`; the same qualitative bias holds under
  Laya's full reward (log + spherical − RPS) and across a range of `K` and
  target entropies, not just the one hand-picked 3-class example in
  `docs/PLAN.md` §4.
- **Config:** none (self-contained script, no YAML) — see script header for
  seeds/steps/batch.
- **Command:**

  ```bash
  uv run python experiments/e1_toy_bias.py
  ```

- **Result:** Confirmed on all three sub-experiments, logged to
  `results/e1_toy_bias.json`.
  - `baseline_replication` (log-score only, `t=[0.7,0.2,0.1]`) exactly
    reproduces the seed-0 table already in `docs/PLAN.md` §4: `p*` = [0.700,
    0.200, 0.100] / [0.723, 0.187, 0.090] / [0.780, 0.154, 0.066] / [0.905,
    0.071, 0.024] at σ = 0 / 0.5 / 1.0 / 2.0.
  - `full_reward` (Laya's log+spherical(−RPS), same target, both `choice`
    and `score` question types, spherical weight 0.5 — see 2026-09-24
    correction below): same monotonic sharpening, slightly stronger than
    log-score-only at matched σ (σ=2.0 gives max_p≈0.923 for `choice` and
    ≈0.932 for `score`, vs. 0.905 for log-only).
  - `k_sweep` (K ∈ {2,5,20,77}, peaked and near-uniform targets, log-only
    and full reward): KL(target‖p*) and `max_p − target_max_p` increase
    monotonically with σ in all 16 (K, target-entropy, reward) configurations tested (64 records: 16 × 4 σ values); the
    effect is largest for peaked, mid-cardinality targets (e.g. K=5 peaked:
    KL rises from 0 to 0.155 log-only / 0.203 full-reward between σ=0 and
    σ=2) and smallest for near-uniform, high-K targets (K=77 near-uniform:
    KL only reaches 0.003–0.008 at σ=2) — consistent with the Step 4
    argument in `docs/analysis/2026-09-24/proof_sketch_smoothed_log_score.md` that the
    bias scales with the curvature of softmax around the target, which is
    largest for peaked, low-to-mid-K targets.
- **Decision:** Adopt as report evidence for §V's proposition (see
  `docs/analysis/2026-09-24/proof_sketch_smoothed_log_score.md`). Confirms prediction 2
  from `docs/RESEARCH.md` §Method ("raw ECE and fitted T increase
  monotonically with σ") at the toy level; E2 checks whether it holds when
  the RL term is combined with Laya's CE term (weight 1.0) on real data,
  which this toy setup does not include. Follow-up (not yet done): extend
  `full_reward`/`k_sweep` to also sweep `w_rl`/`w_ce` mixtures directly,
  rather than only pure-RL objectives, to preview E2's CE-cancellation
  question before spending GPU time.

### 2026-09-24 — correction: spherical reward weight and Laya source verification

- **What happened:** Cloned the actual Laya repo (`github.com/NandhaKishorM/laya`)
  and read `laya/common.py` directly instead of relying only on the plan's
  condensed excerpt. Found the plan's `proper_reward` spherical weight
  (0.75) does not match the real default (`w_sph=0.5` in
  `laya/common.py::proper_reward`). Fixed the default everywhere it was
  used (`src/modules/loss.py`, `src/pipelines/config.py`,
  `experiments/e1_toy_bias.py`) and re-ran E1's `full_reward`/`k_sweep`
  sub-experiments (numbers above are the corrected ones;
  `baseline_replication` is log-score-only so unaffected).
- **Other source-verification fixes**, same pass: `src/modules/model.py`'s
  `nn.TransformerEncoder` now passes `enable_nested_tensor=False` (matches
  `laya/common.py::DecisionModel`; also silences a PyTorch warning that was
  firing on every run); `src/data.py::build_sequence` now strips literal
  mask-token substrings from instructions/options/state before tokenizing
  and drops any marker pushed past the final length cutoff (both match
  defensive guards in `laya/common.py::build_sequence` that the plan's
  condensed version omitted).
- **Known, deliberate gap:** `DecisionModel` still omits Laya's `act_head`
  (a secondary action/escalation head trained with weight 0.0, reported
  AUROC 0.30 — near-useless by Laya's own account) — a checkpoint trained
  here will not load against `convaiinnovations/laya-typed-decisions`'s full
  head, only its shared encoder.
- **Decision:** Adopt the corrected default and guards. Re-verified: 44
  tests pass, `ruff check .` clean, and a 20-case end-to-end pipeline run
  (`--max_examples 20`) still trains/validates/checkpoints/evaluates
  correctly after the fixes.

### 2026-09-25 — pre-E2 verification against Laya's fine-tune notebook

- **What happened:** Read `notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`
  (the run that produced `convaiinnovations/laya-typed-decisions`) and
  inspected `convaiinnovations/laya`'s `model.safetensors` before spending
  GPU budget on E2.
- **Bug fixed — RL advantage normalisation.** The port divided the
  group-centred advantage by `r.std()` (std of *raw* rewards over all
  noise samples and rows, i.e. including between-row reward spread). The
  notebook divides by the std of the *centred* advantages. On a two-row
  example the old version's RL gradient was ~90× smaller (0.0036 vs 0.32),
  so any E2 run on the old code would have under-weighted the very term
  being ablated. Fixed in `src/modules/loss.py`; regression test
  `tests/test_loss.py::test_rl_advantage_normalised_by_centred_std`.
- **Partial reversal of the 2026-09-24 correction.** `laya/common.py`'s
  `proper_reward` default is `w_sph=0.5`, but the notebook calls it with
  `w_sph=0.75`. E2 reproduces the notebook, so `configs/e2/*.yaml` use
  0.75. E1's numbers (0.5) stand as a study of the library default.
- **Evaluation now matches Laya's definition of correct:** accuracy and ECE
  score against the gold hard `label` (differs from the soft target's
  argmax on 85/8000 rows), plus soft accuracy, score MAE and per-type
  breakdowns in `test_eval.json`.
- **Other fixes found on the way:** cosine LR `t_max` was 10 *steps* in
  `configs/train.yaml` (the scheduler steps per optimizer step, so LR
  cycled ~34×/epoch) — now `t_max: auto` = total steps; `amp` was never
  read — now bf16/fp16 autocast (bf16 also on MPS); only `torch` was seeded
  — now `random`/`numpy` too.
- **New for E2/E3:** `init_from` loads Laya's full `DecisionModel` weights
  (tensor names match; only `act_head.*` and the scalar `temperature` are
  dropped — verified on CPU: loads 201 tensors, base Laya on 60 test rows
  gives acc 0.52 at mean confidence 0.68, raw ECE 0.215 — already
  over-confident before any fine-tuning); `lr_head` (Laya: 2.5e-5 encoder /
  1e-4 head); `grad_clip`; `log_grad_diagnostics` (E3 cosine/norm ratio
  w.r.t. logits). A 20-case tiny-encoder dry run of `configs/e2/laya_rlce.yaml`
  already shows ‖∇RL‖/‖∇CE‖ ≈ 7.5 at σ=0.4.
- **Also noted:** Laya's own fitted per-type temperatures on
  `convaiinnovations/laya` are 1.64 / 1.25 / 1.98 (choice / score / noul) —
  all > 1, i.e. over-confident raw outputs, consistent with the §V claim.
- **Laya notebook setup reproduced in `configs/e2/`:** 4 epochs, effective
  batch 64, AdamW wd 0.01, cosine to 1e-6, grad clip 1.0, G=4, σ 0.4→0.1,
  sequences at 512/192. Deliberate differences: bf16 instead of
  fp16+GradScaler, case-level 10% calibration split (Laya: 10% of question
  rows, ≤400), temperature per (type, K-bucket).
- **Compute:** local Apple M5 16 GB can train ModernBERT-large at batch 2–4
  bf16 (~1.7–2.6 rows/s, 10–11 GB), i.e. ~2.5–3 h per 4-epoch run if memory
  is free; a rented A100 is the plan (`scripts/e2/README.md`).
- **Decision:** 54 tests pass (10 added), `ruff` clean. E2 is ready
  to run: `scripts/e2/run_min.sh` on one rented GPU.

### 2026-09-25 — E2 + E3: loss ablation, 13 runs on one rented A100

- **Hypothesis:** `docs/PLAN.md` §5 predictions 1–3: CE-only is at least as
  calibrated as RL+CE; the bias grows with σ; RL-only ends over-confident.
- **Config:** `configs/e2/{laya_rlce,ce_only,rl_only,rlce_sigma0p5_fixed,rlce_sigma1_fixed,rlce_sigma2_fixed}.yaml`.
  Seeds 42/43/44 for laya_rlce, ce_only and σ = 1; seed 42 for the rest;
  plus a seed-42 repeat.
- **Command:** `CONFIGS="laya_rlce:43 ce_only:43 …" HF_REPO=minhleduc/rlcd-e2-checkpoints DELETE_CKPTS=1 bash scripts/e2/run_min.sh`,
  then `scripts/e2/dump_logits.py`; tables from `scripts/e2/summarize.py`.
- **Result:** full write-up in `docs/reports/2026-09-25/e2_minimal_runs.md`.
  - Hard-label ECE has the wrong sign on this dataset (the teacher itself
    scores 0.326; `docs/analysis/2026-09-25/ece_hard_label_vs_soft_target.md`),
    so the claim is tested with sharpness vs targets, fitted T and
    soft-target NLL/Brier.
  - **Prediction 2 confirmed:** RL+CE at fixed σ 0.5 / 1 / 2 gives sharpness
    vs target +0.001 / +0.025 / +0.060, fitted T (noul) 1.22 / 1.39 / 1.72,
    and NLL 0.870 / 0.890 / 0.932. The trend is monotone and far beyond the
    seed spread.
  - **Prediction 1 confirmed on proper scores:** at Laya's own annealed σ,
    CE-only beats RL+CE on NLL (0.861 vs 0.866, every seed), Brier and
    accuracy (0.782 vs 0.773), raw and after temperature; sharpness is
    equal. The RL term adds nothing CE doesn't.
  - **Prediction 3 partly refuted:** RL-only (1 seed) is less accurate
    (0.760) but not over-confident at σ ≤ 0.4.
  - **E3:** at Laya's schedule ‖∇RL‖/‖∇CE‖ = 7–110 with cos ≈ 0.6–0.7; in the
    fixed-σ runs ‖∇RL‖ ∝ 1/σ and cos falls 0.60 → 0.52 → 0.37.
  - The Laya recipe reproduces (acc 0.773, Brier 0.054 vs README 0.766 /
    0.062). The seed-42 repeat reproduces closely. One earlier seed-42
    attempt diverged (acc 0.697); it is kept, excluded and explained in the
    report.
- **Decision:** Adopt as report §VI evidence, with the metric change
  (lead with fitted T and soft-target proper scores; hard-label ECE shown
  next to the teacher's 0.326). Follow-ups: 2 more seeds for σ = 0.5/2 and
  RL-only; offline per-row sharpness analysis from `logits.pt`; E5 on
  these checkpoints.
