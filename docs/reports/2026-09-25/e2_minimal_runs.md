# E2 / E3: loss ablation on `LocalLLaMA/typed-decisions`, 13 runs

Date: 2026-09-25. Machine: 1× A100-SXM4-40GB rented on ckey.vn (see
`compute_evaluation.md`). Total cost about **36,000 VND** (~$1.4) for about
2.4 h of VM time, including setup, a lost first attempt and the logit dump.
Interpretation depends on
`docs/analysis/2026-09-25/ece_hard_label_vs_soft_target.md`: on this
dataset **hard-label ECE has the wrong sign** for the over-confidence claim.
The claim is tested with sharpness relative to the soft targets, fitted T,
and soft-target NLL/Brier.

## Setup

- Every run is a single-GPU replica of Laya's typed-decisions fine-tune
  (`configs/e2/laya_rlce.yaml`; parity table in
  `docs/analysis/2026-09-25/laya_source_verification.md` §6). All runs start
  from `convaiinnovations/laya` and train for a fixed 4 epochs; the final
  weights are evaluated.
- Arms (`configs/e2/`): `laya_rlce` (w_rl = w_ce = 1, σ 0.4→0.1 annealed),
  `ce_only` (w_rl = 0), `rl_only` (w_ce = 0), and `rlce_sigma{0p5,1,2}_fixed`
  (RL+CE with σ held constant).
- Seeds: 42/43/44 for `laya_rlce`, `ce_only` and σ = 1; seed 42 only for
  `rl_only`, σ = 0.5 and σ = 2. `laya_rlce` seed 42 was run a second time as
  a reproducibility check (`_rep2`).
- Test: 2,000 question rows. Temperature is fitted per (type, K-bucket) on
  the run's own held-out 10% calibration cases.
- Metrics come from `test_eval_v2.json`, re-scored from the saved weights by
  `scripts/e2/dump_logits.py` with bf16 inference. These match the
  training-time fp32 `test_eval.json` to within ±0.002 ECE. Tables are built
  by `scripts/e2/summarize.py` → `results/e2_summary.json`.

## Main table (mean ± std over seeds)

`conf − t_max` = mean max-prob of the model minus that of the soft target
(target mean max = 0.659). **> 0 means sharper than the targets.** T is the
fitted temperature for the dominant bucket of each type.

| arm | seeds | conf − t_max | NLL vs t | Brier vs t | post-T NLL | T choice | T noul | T score | hard-label raw ECE | post-T ECE | accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **ce_only** | 3 | −0.016 ± 0.004 | **0.861 ± 0.001** | **0.052 ± 0.001** | **0.857** | 1.12 ± 0.05 | 1.10 ± 0.03 | 1.08 ± 0.01 | 0.139 ± 0.001 | 0.158 | **0.782 ± 0.004** |
| laya_rlce (Laya default) | 3 | −0.017 ± 0.004 | 0.866 ± 0.002 | 0.054 ± 0.001 | 0.863 | 1.09 ± 0.02 | 1.10 ± 0.01 | 1.06 ± 0.04 | 0.132 ± 0.003 | 0.146 | 0.773 ± 0.002 |
| rl_only | 1 | −0.024 | 0.868 | 0.055 | 0.866 | 1.08 | 1.15 | 1.06 | 0.127 | 0.141 | 0.760 |
| RL+CE, σ = 0.5 fixed | 1 | +0.001 | 0.870 | 0.057 | 0.863 | 1.15 | 1.22 | 1.12 | 0.114 | 0.144 | 0.774 |
| RL+CE, σ = 1 fixed | 3 | +0.025 ± 0.006 | 0.890 ± 0.003 | 0.065 ± 0.001 | 0.866 | 1.28 ± 0.03 | 1.39 ± 0.05 | 1.29 ± 0.02 | 0.083 ± 0.004 | 0.137 | 0.764 ± 0.005 |
| RL+CE, σ = 2 fixed | 1 | **+0.060** | 0.932 | 0.075 | 0.865 | **1.49** | **1.72** | **1.57** | 0.056 | 0.145 | 0.774 |
| *reference:* teacher `t` as predictor | — | 0 | — | 0 | — | — | — | — | 0.326 | — | 0.984 |
| *reference:* Laya README (fine-tuned) | — | — | — | 0.062 | — | — | — | — | — | 0.213 | 0.766 |

Per-run rows are in `results/e2_summary.json` and in the printout of
`scripts/e2/summarize.py`.

## Predictions from `the project plan` §5

**Prediction 2 (bias grows with σ): confirmed**, on the metrics that test
it. As σ goes 0.5 → 1 → 2 (RL+CE, fixed):
- sharpness vs target rises +0.001 → +0.025 → +0.060,
- fitted T rises in every question type (choice 1.15 → 1.28 → 1.49, noul
  1.22 → 1.39 → 1.72, score 1.12 → 1.29 → 1.57),
- soft-target NLL worsens 0.870 → 0.890 → 0.932, and Brier 0.057 → 0.065 →
  0.075.

The trend is monotone at every step and far larger than the seed spread at
σ = 1 (3 seeds). This is the §V proposition observed at full scale on a
421M-parameter model, and it matches E1's toy result. As the analysis note
predicts, hard-label raw ECE *falls* with σ (0.114 → 0.083 → 0.056). The
reason is that sharpening moves confidence toward hard-label accuracy, not
that calibration improves.

**Prediction 1 (CE-only ≤ RL+CE): confirmed on proper scores; as phrased
(hard-label ECE), not a valid test.** At Laya's own schedule (σ annealed
0.4 → 0.1), the RL term changes sharpness by nothing measurable
(−0.016 vs −0.017) and the fitted T values overlap. CE-only is better on
every soft-target proper score, both raw and after temperature (NLL 0.861
vs 0.866; post-T NLL 0.857 vs 0.863; Brier 0.052 vs 0.054), and it is more
accurate (0.782 vs 0.773). The gaps are small but exceed the seed spread
consistently: every CE-only seed beats every RL+CE seed on NLL. Laya's RL
term therefore adds nothing that CE does not already provide, and costs a
little. RL+CE's lower hard-label ECE (0.132 vs 0.139) points the opposite
way for the reason explained above.

**Prediction 3 (RL-only converges slower and ends over-confident):
partly refuted.** With one seed, RL-only ends *less* accurate (0.760) and
with worse NLL (0.868) than either arm with CE. It is **not** over-confident
at the annealed σ (−0.024 vs target). With σ ≤ 0.4 and the advantage
normalisation, the smoothing bias is too small to show here, even without
CE. Caveat: every arm starts from an already fine-tuned Laya checkpoint,
which favours RL-only.

## E3: gradient diagnostics (w.r.t. the logits, every 10 steps)

| σ | runs | cos(∇RL, ∇CE) | ‖∇RL‖ / ‖∇CE‖ |
|---|---|---|---|
| 0.4 → 0.1 (annealed) | laya_rlce ×4, ce_only ×3, rl_only | 0.58–0.74 per epoch | 7–10 (σ = 0.4) → 64–113 (σ = 0.1) |
| 0.5 fixed | 1 | 0.60 ± 0.13 | 10.4 |
| 1 fixed | 3 | 0.52 ± 0.14 | 4.4–5.4 |
| 2 fixed | 1 | 0.37 ± 0.20 | 2.2 |

- In the fixed-σ runs, where σ is not confounded with training time, the
  RL/CE norm ratio times σ is about constant (5.2, 4.9, 4.3). So ‖∇RL‖ ∝
  1/σ, as expected for `adv · ε / σ²` with O(1) normalised advantages
  (`docs/analysis/2026-09-25/laya_source_verification.md` §1).
- **At Laya's own schedule the RL term is 7–110× larger than the CE term**
  and only moderately aligned with it (cos ≈ 0.6–0.7). Before grad-norm
  clipping (1.0), the RL estimate dominates the update direction. The final
  models nonetheless match CE-only closely, consistent with the RL estimate
  being a noisy but roughly unbiased proxy for the CE direction at small σ.
- Alignment falls as σ grows (0.60 → 0.52 → 0.37). This matches the
  smoothed objective's gradient drifting away from the unsmoothed CE
  gradient (§V.1).
- Diagnostics are logged in `ce_only` too (computed, weight 0). The
  annealed-arm ratios rise as σ falls partly because ‖∇CE‖ shrinks as
  training converges; use the fixed-σ rows for scaling claims.

## Reproducibility

- `laya_rlce` seed 42 run twice on the final code: raw ECE 0.129 / 0.131,
  NLL 0.864 / 0.867, accuracy 0.772 / 0.772. Identical settings reproduce
  closely but not bitwise (GPU nondeterminism).
- **Excluded outlier:** the very first `laya_rlce` seed-42 run (the first
  training job on the fresh VM; `results/e2_vm/results/_attempt1/`) matched
  the later runs exactly at step 1 (loss 1.4084) but then diverged. It ended
  at accuracy 0.697, NLL 0.936, Brier 0.096 and raw ECE 0.110. Between it
  and the rep runs only the W&B logging code changed, which does not touch
  training. The cause is unknown; a likely suspect is the first-ever
  `torch.compile`/Triton build (gcc was installed minutes before). It is
  kept on disk and reported here rather than silently dropped. Its
  `ce_only` companion was stopped mid-run.

## Laya reproduction check

Our RL+CE (Laya recipe) gives accuracy 0.773 ± 0.002 and Brier 0.054 ± 0.001.
Laya's README reports 0.766 and 0.062 for its 2×T4 run. The reproduction is
faithful, and slightly better on one A100 with bf16.

## Artifacts

| what | where |
|---|---|
| weights (fp32 safetensors, 1.6 GB each), per-run JSONs, raw calib/test logits (`logits.pt`) | HF `minhleduc/rlcd-e2-checkpoints` (private); local copy of weights in `checkpoints/e2/` |
| training curves, E3 diagnostics, test summaries | W&B `octoopt/rlcd-reverse-engineering`, group `e2-minimal` |
| run logs, `pip freeze`, GPU info, first attempt | `results/e2_vm/` (git-ignored) |
| tables | `results/e2_summary.json` via `scripts/e2/summarize.py --runs_dir results/e2_vm/results` |

## Open items

- σ = 0.5, σ = 2 and RL-only have one seed each. Two more seeds each
  (6 runs, ~50 min on an A100, ~13,000 VND) would complete the 3-seed matrix.
- A per-row sharpness analysis (by type, K and target entropy, as in E1's
  k_sweep) can be done offline from the dumped `logits.pt`, with no GPU.
- E4 (reward composition) and E5 (consistency probes) are not started. E5
  is inference-only and can reuse these checkpoints.
