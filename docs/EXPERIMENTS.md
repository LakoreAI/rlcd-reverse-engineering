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
    monotonically with σ in every one of the 64 configurations tested; the
    effect is largest for peaked, mid-cardinality targets (e.g. K=5 peaked:
    KL rises from 0 to 0.155 log-only / 0.203 full-reward between σ=0 and
    σ=2) and smallest for near-uniform, high-K targets (K=77 near-uniform:
    KL only reaches 0.003–0.008 at σ=2) — consistent with the Step 4
    argument in `docs/analysis/proof_sketch_smoothed_log_score.md` that the
    bias scales with the curvature of softmax around the target, which is
    largest for peaked, low-to-mid-K targets.
- **Decision:** Adopt as report evidence for §V's proposition (see
  `docs/analysis/proof_sketch_smoothed_log_score.md`). Confirms prediction 2
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
