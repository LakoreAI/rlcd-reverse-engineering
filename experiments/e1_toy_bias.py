"""E1 — toy verification that the noise-smoothed proper score has a
systematically over-confident optimum.

Claim under test: minimising E_eps[R(softmax(z + eps))], eps ~ N(0, sigma^2 I)
projected to zero-mean over the valid options, drives the *noise-free*
softmax(z*) sharper than the target distribution t, and the effect grows
with sigma. At sigma=0 this recovers the ordinary log-score optimum
softmax(z*) = t exactly.

Three sub-experiments, each written to results/e1_toy_bias.json:

  1. baseline_replication — a seed-0 sigma sweep on t=[0.7,0.2,0.1] with the
     log-score-only objective, as a hand-checkable sanity case.
  2. full_reward — same target, but with Laya's full reward (log score +
     0.5*spherical - 1.0*RPS for score-type questions, matching
     `laya/common.py::proper_reward`'s actual defaults) instead of the log
     score alone, for both a "choice" question (no RPS) and a "score"
     question (RPS active, target is ordinal).
  3. k_sweep — repeats (1) and (2) across K in {2, 5, 20, 77} and across
     target distributions of varying entropy, to check the bias trend holds
     beyond the one hand-picked 3-class example.

We use torch autograd for the smoothed-objective gradient (reparameterising
through z + eps, which is differentiable) rather than a REINFORCE/score-
function estimator — that estimator's own variance is E3's question, not
this one's. E1 only asks: what is the smoothed objective's true optimum?
"""

import json
from pathlib import Path

import torch

RESULTS_PATH = Path(__file__).resolve().parents[1] / "results" / "e1_toy_bias.json"

SEED = 0
STEPS = 4000
BATCH = 4096
LR = 0.05


def proper_reward(
    q: torch.Tensor,
    t: torch.Tensor,
    is_score_type: bool,
    w_sph: float = 0.5,
    w_rps: float = 1.0,
) -> torch.Tensor:
    """Laya's reward, matching `laya/common.py::proper_reward`'s actual
    defaults: log score + w_sph * spherical score - w_rps * RPS (RPS only
    for score-type / ordinal questions). q, t: (..., K).
    """
    log_s = (t * torch.log(q.clamp_min(1e-12)).clamp_min(-9.21)).sum(-1)
    sph = (t * q).sum(-1) / q.norm(dim=-1).clamp_min(1e-9)
    reward = log_s + w_sph * sph
    if is_score_type:
        k = torch.tensor(float(q.shape[-1])).clamp(min=2)
        rps = ((q.cumsum(-1) - t.cumsum(-1)) ** 2).sum(-1) / (k - 1)
        reward = reward - w_rps * rps
    return reward


def smoothed_optimum(
    t: torch.Tensor,
    sigma: float,
    reward: str = "log_only",
    is_score_type: bool = False,
    steps: int = STEPS,
    batch: int = BATCH,
    lr: float = LR,
    seed: int = SEED,
    return_logits: bool = False,
) -> torch.Tensor:
    """Gradient-descend z to minimise E_eps[-R(softmax(z+eps))] (or the pure
    log score when reward="log_only") and return the noise-free softmax(z*).
    """
    g = torch.Generator().manual_seed(seed)
    k = t.shape[-1]
    z = torch.zeros(k, requires_grad=True)
    opt = torch.optim.Adam([z], lr=lr)

    for _ in range(steps):
        eps = torch.randn((batch, k), generator=g) * sigma
        eps = eps - eps.mean(-1, keepdim=True)  # zero-mean projected noise
        zs = z + eps

        if reward == "log_only":
            loss = -(t * torch.log_softmax(zs, dim=-1)).sum(-1).mean()
        elif reward == "full":
            q = torch.softmax(zs, dim=-1)
            loss = -proper_reward(q, t, is_score_type).mean()
        else:
            raise ValueError(f"unknown reward mode: {reward!r}")

        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        if return_logits:
            return z.detach()
        return torch.softmax(z, dim=-1)


def noise_average(
    z: torch.Tensor, sigma: float, samples: int = BATCH, seed: int = SEED
):
    """E_eps[softmax(z + eps)] with eps ~ N(0, sigma^2) projected to zero mean
    — what the smoothed objective (Eq. 5) guarantees matches the target,
    in contrast to the noise-free softmax(z) used at inference."""
    g = torch.Generator().manual_seed(seed)
    k = z.shape[-1]
    eps = torch.randn((samples, k), generator=g) * sigma
    eps = eps - eps.mean(-1, keepdim=True)
    return torch.softmax(z.unsqueeze(0) + eps, dim=-1).mean(0)


def kl(t: torch.Tensor, p: torch.Tensor) -> float:
    return float((t * (t.clamp_min(1e-12).log() - p.clamp_min(1e-12).log())).sum())


def sharpness(p: torch.Tensor) -> float:
    """max probability — a scalar proxy for how peaked/over-confident p is."""
    return float(p.max())


def run_baseline_replication() -> list[dict]:
    """A seed-0 sigma sweep on t=[0.7, 0.2, 0.1] with the log-score-only
    objective.
    """
    t = torch.tensor([0.7, 0.2, 0.1])
    records = []
    for sigma in (0.0, 0.5, 1.0, 2.0):
        z = smoothed_optimum(t, sigma, reward="log_only", return_logits=True)
        p = torch.softmax(z, dim=-1)
        pbar = noise_average(z, sigma) if sigma > 0 else p
        records.append(
            {
                "sigma": sigma,
                "target": t.tolist(),
                "noise_free_p": [round(v, 3) for v in p.tolist()],
                "noise_averaged_p": [round(v, 3) for v in pbar.tolist()],
                "kl_p_from_t": kl(t, p),
                "kl_noise_averaged_from_t": kl(t, pbar),
                "sharpness_max_p": sharpness(p),
                "sharpness_max_noise_averaged": sharpness(pbar),
            }
        )
        print(
            f"[baseline] sigma={sigma}: p*={p.round(decimals=3).tolist()} "
            f"E[p]={pbar.round(decimals=3).tolist()} target={t.tolist()}"
        )
    return records


def run_full_reward() -> list[dict]:
    """Same target as the baseline, but under Laya's full reward, once as a
    'choice' question (no RPS) and once as a 'score' question (RPS active).
    """
    t = torch.tensor([0.7, 0.2, 0.1])
    records = []
    for qtype, is_score_type in (("choice", False), ("score", True)):
        for sigma in (0.0, 0.5, 1.0, 2.0):
            z = smoothed_optimum(
                t, sigma, reward="full", is_score_type=is_score_type, return_logits=True
            )
            p = torch.softmax(z, dim=-1)
            pbar = noise_average(z, sigma) if sigma > 0 else p
            records.append(
                {
                    "qtype": qtype,
                    "sigma": sigma,
                    "target": t.tolist(),
                    "noise_free_p": [round(v, 3) for v in p.tolist()],
                    "noise_averaged_p": [round(v, 3) for v in pbar.tolist()],
                    "kl_p_from_t": kl(t, p),
                    "kl_noise_averaged_from_t": kl(t, pbar),
                    "sharpness_max_p": sharpness(p),
                    "sharpness_max_noise_averaged": sharpness(pbar),
                }
            )
            print(
                f"[full_reward:{qtype}] sigma={sigma}: "
                f"p*={p.round(decimals=3).tolist()} E[p]={pbar.round(decimals=3).tolist()} "
                f"target={t.tolist()}"
            )
    return records


def _dirichlet_sample(concentration: torch.Tensor, seed: int) -> torch.Tensor:
    # torch.distributions.Gamma draws from the global RNG, so reseed it
    # explicitly per call rather than threading a Generator through, to keep
    # each target draw reproducible regardless of call order.
    torch.manual_seed(seed)
    gammas = torch.empty_like(concentration)
    for i, c in enumerate(concentration.tolist()):
        gammas[i] = torch.distributions.Gamma(c, 1.0).sample()
    return gammas / gammas.sum()


def run_k_sweep() -> list[dict]:
    """K in {2, 5, 20, 77} (77 matches Banking77's class count, one of the
    datasets Laya's README reports failing on at that cardinality), each
    with a low-entropy (peaked, concentration=0.3) and a high-entropy
    (near-uniform, concentration=5.0) target, log-score-only and full-reward
    (choice), across the same sigma sweep.
    """
    records = []
    for k in (2, 5, 20, 77):
        for label, concentration in (("peaked", 0.3), ("near_uniform", 5.0)):
            conc = torch.full((k,), concentration)
            t = _dirichlet_sample(conc, seed=SEED + k)
            t = t.clamp_min(1e-4)
            t = t / t.sum()
            for reward in ("log_only", "full"):
                for sigma in (0.0, 0.5, 1.0, 2.0):
                    p = smoothed_optimum(
                        t,
                        sigma,
                        reward=reward,
                        is_score_type=False,
                        steps=2000,
                        batch=2048,
                    )
                    records.append(
                        {
                            "k": k,
                            "target_entropy_label": label,
                            "reward": reward,
                            "sigma": sigma,
                            "target_entropy_nats": float(
                                -(t * t.clamp_min(1e-12).log()).sum()
                            ),
                            "kl_p_from_t": kl(t, p),
                            "sharpness_max_p": sharpness(p),
                            "target_sharpness_max_t": sharpness(t),
                        }
                    )
                    print(
                        f"[k_sweep] k={k:3d} target={label:12s} reward={reward:9s} "
                        f"sigma={sigma}: KL={kl(t, p):.4f} max_p={sharpness(p):.3f} "
                        f"(target max_t={sharpness(t):.3f})"
                    )
    return records


def main(out_path: Path | None = None) -> dict:
    torch.manual_seed(SEED)
    results = {
        "baseline_replication": run_baseline_replication(),
        "full_reward": run_full_reward(),
        "k_sweep": run_k_sweep(),
    }
    out_path = out_path or RESULTS_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved: {out_path}")
    return results


if __name__ == "__main__":
    main()
