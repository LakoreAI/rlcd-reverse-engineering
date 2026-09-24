# Proof sketch: the noise-smoothed log score has an over-confident optimum

Status: sketch for report §V (Analysis of the RL term), backing the E1
results in `results/e1_toy_bias.json`. Covers the log-score case only; the
full Laya reward (log + spherical − RPS) is checked numerically in E1's
`full_reward` / `k_sweep` runs, not proved here.

## Setup

Fix a target distribution `t ∈ Δ^{K-1}` and logits `z ∈ R^K`. Let
`eps ~ N(0, σ² I)` be noise, optionally projected to the zero-mean
hyperplane (Laya's noise is projected this way; the argument below does not
need the projection). Define the smoothed log-score objective

```
J_σ(z) = E_eps[ Σ_i t_i · log softmax(z + eps)_i ]
```

Laya's training loop maximises (a Monte-Carlo, score-function estimate of)
`J_σ`; at inference time the model reports the **noise-free** distribution
`p(z) = softmax(z)`. The question is how the maximiser `z*_σ = argmax_z
J_σ(z)` compares to the maximiser at `σ = 0`, which is exactly `p(z*_0) = t`
(the ordinary cross-entropy optimum).

## Claim

`p(z*_σ)` is at least as sharp as `t` for every `σ ≥ 0`, in the sense that
`E_eps[softmax(z*_σ + eps)] = t` while `softmax` is a strictly concave map
on each coordinate's exposure to noise near the optimum — so the noise-free
point `softmax(z*_σ)` sits on the "sharp" side of the noise-averaged one.
Concretely: `max_i p(z*_σ)_i ≥ max_i t_i`, with strict inequality for `σ >
0` whenever `t` is not already a vertex of the simplex, and the gap grows
with `σ`.

## Argument

**Step 1 — stationarity condition.** `J_σ` is concave in `z` up to the usual
softmax invariances (shift-invariance), because `log softmax(z+eps)_i` is
concave in `z` for each `eps`, and expectation preserves concavity. So a
stationary point of `J_σ` is a global maximiser. The stationarity condition
`∇_z J_σ(z*) = 0` works out to

```
E_eps[ softmax(z*_σ + eps) ] = t.
```

(Differentiating the log-score term gives `∇_z Σ_i t_i log q_i = t - q`
pointwise, where `q = softmax(z+eps)`; taking expectation and setting to
zero yields the display above. This is the direct, `σ=0`-limit-consistent
generalisation of the familiar cross-entropy stationarity condition
`softmax(z*) = t`.)

**Step 2 — Jensen / concavity of the noise average.** `softmax` is not a
linear map of its input, so in general `E_eps[softmax(z+eps)] ≠
softmax(E_eps[z+eps]) = softmax(z)`. Write `p_σ = softmax(z*_σ)` for the
noise-free distribution at the smoothed optimum. Because `softmax_i` is a
smooth, strictly log-convex function of the *other* coordinates' logits
relative to `i` (equivalently: `log-sum-exp` is convex, and `softmax_i(x) =
exp(x_i) / Σ_j exp(x_j)`, whose denominator is convex in `x`), Jensen's
inequality applied to the convex map `x ↦ exp(x_i)` and the concave
reciprocal-of-sum term together imply that averaging softmax over
symmetric, zero-mean logit perturbations *flattens* the output relative to
the noise-free evaluation at the mean logits, for logits away from the
all-equal point. Formally, for the two-coordinate reduction (WLOG compare
class `i` against the pooled rest), softmax reduces to a logistic sigmoid
`σ(z_i - z_rest)`, which is concave for `z_i > z_rest` and convex for
`z_i < z_rest`; averaging a concave-then-convex function of symmetric noise
pulls the *large* coordinate's expectation **down** and the *small*
coordinate's expectation **up** relative to the noise-free value at the
mean argument — i.e. `E_eps[softmax(z+eps)]` is a flattened (higher-entropy,
closer to uniform) version of `softmax(z)`, coordinatewise, for the same
`z`.

**Step 3 — invert the relationship.** Step 1 says the *flattened* quantity
`E_eps[softmax(z*_σ + eps)]` equals `t` exactly. Step 2 says the flattened
quantity is less sharp than the noise-free `softmax(z*_σ)` evaluated at the
same logits. Combining: `softmax(z*_σ)` (the report/inference-time output)
must be *sharper* than `t` — because it is the un-flattened point whose
flattened version *is* `t`. In symbols, if `flatten_σ(·)` denotes the
(coordinatewise entropy-increasing) noise-averaging map from Step 2, then
`flatten_σ(p_σ) = t` and `flatten_σ` strictly increases entropy (decreases
`max_i`) away from the uniform point, so `p_σ` must have *lower* entropy
(higher `max_i`) than `t`, i.e. `max_i p_σ,i ≥ max_i t_i`.

**Step 4 — monotonicity in σ.** `flatten_σ` is the identity at `σ = 0` and
its entropy-increasing effect (the curvature term in Step 2) scales with
`σ²` to leading order (a standard second-order Taylor/Laplace expansion of
`E_eps[softmax(z+eps)]` around `eps=0` picks up a `+ (σ²/2)·Hessian` term
whose sign flattens the peak coordinate). So the amount of "un-flattening"
needed to keep `flatten_σ(p_σ) = t` fixed grows with `σ`, giving `p_σ`
monotonically sharper as `σ` increases. This matches the `sigma ∈
{0, 0.5, 1, 2}` trend in `results/e1_toy_bias.json` (baseline_replication
and k_sweep records) at every `K` and every target-entropy level tested.

## What this does and doesn't say

- This is a **sketch**, not a rigorous proof: Step 2's two-coordinate
  reduction and Step 4's small-`σ` expansion are the informal parts; a
  full proof needs either a majorization argument for the exact softmax
  (not just a pairwise reduction) or a strict verification that the
  Hessian term in Step 4 has the claimed sign for all `K` and all interior
  `t` (not just numerically, as E1 checks).
- It only covers the **pure log-score** term. Laya's actual reward adds a
  spherical term and (for score-type questions) subtracts an RPS term;
  E1's `full_reward` run checks numerically that the same qualitative bias
  survives under the full reward, but the argument above does not extend
  analytically to it.
- It says nothing about the **score-function/ES gradient estimator**'s own
  variance or bias as an estimator of `∇J_σ` — that is E3's question. This
  sketch is entirely about the smoothed **objective**'s optimum, assuming
  it is optimised exactly (E1 uses reparameterised/autodiff gradients of
  `J_σ` for this reason, not the ES estimator).
- Laya's CE term (weight 1.0, alongside the RL term) directly optimises the
  unsmoothed log score and pulls the combined optimum back toward `t`; the
  net calibration effect is a function of `w_rl / w_ce` and the σ schedule,
  not of the RL term in isolation. Laya's reported raw ECE (0.466) is
  therefore evidence that, empirically, the RL term's bias is not fully
  cancelled by the CE term at the weights/schedule Laya uses.
