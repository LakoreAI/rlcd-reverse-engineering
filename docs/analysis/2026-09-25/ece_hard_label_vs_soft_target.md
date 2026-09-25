# Hard-label ECE is the wrong lens for E2 on `LocalLLaMA/typed-decisions`

Date: 2026-09-25. Status: finding from the first E2 runs. It changes which
metric the report uses to test the §V over-confidence claim.

## Observation

The first three E2 runs (seed 42; tables in
`docs/reports/2026-09-25/e2_minimal_runs.md`) are all **under-confident
relative to the gold hard labels**. Their mean max-probability is 0.08–0.14
below accuracy. Yet their temperatures, fitted by NLL against the soft
targets, are **T > 1**, which means the raw outputs are *sharper* than the
targets. Temperature scaling then makes hard-label ECE **worse** in every
run. The run trained with the strongest noise (σ = 1 fixed) has the
highest T (1.26–1.37) and also the *lowest* raw ECE (0.078).

## Explanation: the labels are the teacher's argmax, not outcomes

Treat the dataset's own soft targets `t` as a predictor on the 2,000 test
rows and score it the way E2 scores models (`scripts/e2/summarize.py`
conventions, 15-bin ECE on max-probability):

| predictor | mean max-prob | accuracy vs gold `label` | ECE vs `label` |
|---|---|---|---|
| teacher targets `t` | 0.659 (choice 0.635, noul 0.771, score 0.593) | **0.984** | **0.326** |

The gold `label` equals `argmax t` on 98.4% of test rows. It is derived
from the teacher distribution rather than drawn as an outcome from it. A
model that reproduces the targets *perfectly* (p = t) would therefore have
raw ECE ≈ 0.33, worse than every E2 run. Against these labels:

- lower hard-label ECE means **sharper than the targets**, and
- the §V bias (sharpening as σ grows) *lowers* hard-label ECE instead of
  raising it.

So `docs/PLAN.md` §5 prediction 1 ("CE-only raw ECE ≤ RL+CE raw ECE") and
the ECE half of prediction 2 have the wrong sign on this dataset. They
cannot test the claim as stated. This does not affect Laya's published
numbers as *comparisons*, but it matters for reading them: Laya's "raw ECE
0.466 → 0.081 after temperature" is also computed against these labels.

## What measures the §V claim here

The claim is about sharpness **relative to the target distribution `t`**.
On this dataset that is measured by:

1. **Fitted temperature T per (type, K-bucket)**, fitted by NLL against `t`
   on the calibration slice. T > 1 means sharper than `t`, and the
   prediction is that T grows with σ. The first runs agree: CE-only
   1.09–1.13, annealed RL+CE 1.01–1.11, fixed σ = 1: 1.26–1.37.
2. **Soft-target proper scores**: NLL against `t` (cross-entropy) and Brier
   against `t`. The prediction is that these get worse with σ. First runs:
   NLL 0.860 / 0.864 / 0.888 and Brier 0.051 / 0.053 / 0.064 (CE-only /
   RL+CE / σ = 1).
3. Optionally, mean max p − mean max t (signed sharpness gap). This can be
   computed from the saved checkpoints without retraining.

Hard-label ECE stays in the report because the plan requires raw ECE next
to post-T ECE. It should be presented with the teacher's own 0.326 as the
reference point, not as the calibration verdict.

## Consequences for the report (§VI)

- Headline E2 table: fitted T and NLL/Brier against soft targets. Hard-label
  ECE and accuracy go in secondary columns, with the teacher reference row.
- Restate prediction 1 as "CE-only fitted T ≤ RL+CE fitted T" and "CE-only
  soft-target NLL ≤ RL+CE". Report the original ECE form as-is and explain
  why it isn't a valid test here.
- A true outcome-calibration test needs labels drawn *from* a distribution
  (human multi-annotator sets such as ChaosNLI; `docs/PLAN.md` §6 future
  work), not a teacher's argmax.
