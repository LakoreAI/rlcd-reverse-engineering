# Per-row sharpness of the E2 models (offline, from dumped logits)

Date: 2026-09-25. Inputs: the `logits.pt` test dumps of all 13 E2 runs
(HF `minhleduc/rlcd-e2-checkpoints`, local `checkpoints/e2/`). Script:
`scripts/e2/row_analysis.py` → `results/e2_row_analysis.json`. No GPU is
needed. All numbers use raw (untempered) logits on the 2,000 test rows,
averaged over seeds where a config has several (the n column).

## Target-referenced ECE

Hard-label ECE cannot test the over-confidence claim on this dataset
(`ece_hard_label_vs_soft_target.md`). The reference point that fits the
§V claim is the **soft target**. Bin rows by the model's max-probability
p_max, and in each bin compare it with **t[argmax p]**, the target's
probability for the option the model picked:

  soft-ECE = Σ_b (n_b / n) · | mean_b(p_max) − mean_b(t[argmax p]) |

This is 0 for a model whose distributions equal the targets. In each bin,
p_max > t[argmax p] means sharper than the target. It is the ordinary ECE
construction with the 0/1 hit indicator replaced by the target
probability, i.e. calibration against a known label distribution rather
than against sampled outcomes.

| config | n | sharpness gap mean(p_max) − mean(t_max) | **soft-ECE** | hard-label ECE |
|---|---|---|---|---|
| teacher `t` itself | — | 0 | **0.000** | 0.326 |
| rl_only | 1 | −0.024 | 0.032 | 0.127 |
| ce_only | 3 | −0.016 | **0.034** | 0.139 |
| laya_rlce (Laya default, σ 0.4→0.1) | 3 | −0.017 | 0.036 | 0.132 |
| RL+CE, σ = 0.5 fixed | 1 | +0.001 | 0.054 | 0.114 |
| RL+CE, σ = 1 fixed | 3 | +0.025 | 0.080 | 0.083 |
| RL+CE, σ = 2 fixed | 1 | +0.060 | **0.111** | 0.056 |

As σ grows, soft-ECE rises monotonically (0.054 → 0.080 → 0.111, 3.3× the
CE-only value at σ = 2). Hard-label ECE *falls* monotonically (0.114 →
0.083 → 0.056) over the same runs. The two metrics rank the same models in
opposite order. This is the paper's headline figure (Fig. 2).

At Laya's own schedule, RL+CE and CE-only are indistinguishable on
soft-ECE (0.036 vs 0.034) and sharpness (−0.017 vs −0.016). RL-only
(1 seed) is marginally lowest on soft-ECE but is also the least accurate
run (0.760).

## By question type (sharpness gap)

| config | choice | score | noul |
|---|---|---|---|
| ce_only | −0.021 | −0.011 | −0.017 |
| laya_rlce | −0.016 | −0.019 | −0.016 |
| σ = 0.5 | +0.003 | +0.000 | −0.001 |
| σ = 1 | +0.028 | +0.033 | +0.010 |
| σ = 2 | +0.065 | +0.066 | +0.048 |

The σ effect appears in every type. It is weakest for `noul` (K = 2), in
line with E1's k_sweep, where the bias is smallest at K = 2 for the same σ.
The fitted temperatures give the opposite ordering (noul has the largest T:
1.72 at σ = 2). The two measures differ because T is fitted by NLL, which
weights confident mistakes heavily. Both agree the effect grows with σ.

## By target entropy (terciles of H(t); means 0.38 / 0.82 / 1.12 nats)

| config | low H (peaked) | mid | high H (flat) |
|---|---|---|---|
| ce_only | −0.056 | −0.017 | +0.026 |
| laya_rlce | −0.059 | −0.017 | +0.025 |
| σ = 0.5 | −0.046 | +0.002 | +0.047 |
| σ = 1 | −0.028 | +0.026 | +0.076 |
| σ = 2 | +0.008 | +0.069 | +0.104 |

Two effects add up:

1. **Regression toward the mean, independent of RL.** Every model,
   CE-only included, is *under*-sharp on peaked targets and *over*-sharp on
   flat ones. A finite model fitted by a proper score on noisy teacher
   distributions shrinks toward typical confidence. The mean sharpness gap
   of CE-only (−0.016) is therefore a balance of two opposite biases, not
   zero bias.
2. **The σ effect is a near-uniform upward shift** of about +0.03 to +0.08
   across terciles from σ = 0.5 to 2, slightly larger for flatter targets.
   In absolute p_max, E1 predicts the largest shift for peaked, mid-K
   targets; here the peaked tercile starts furthest below target, so the
   same shift moves it toward 0. A per-tercile *relative* comparison with
   E1 needs matched (K, H) cells, which is left for the final paper
   revision.

## Caveats

- σ = 0.5, σ = 2 and rl_only have one seed each; the σ = 1 values are
  3-seed means with seed spread ≤ 0.006 on the sharpness gap.
- soft-ECE uses the same 15 equal-width bins as the hard-label ECE, so the
  two columns are directly comparable.
- Temperature scaling removes most of the σ penalty on NLL (0.932 → 0.865
  at σ = 2; `docs/reports/2026-09-25/e2_minimal_runs.md`). The bias is a
  systematic, largely temperature-correctable distortion, consistent with a
  noise-smoothed objective whose optimum is a sharpened version of the
  target.
