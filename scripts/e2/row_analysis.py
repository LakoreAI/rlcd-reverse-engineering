"""Offline per-row analysis of the dumped E2 logits (no GPU needed).

For every `<runs_dir>/<run>/logits.pt` (from dump_logits.py) computes, on the
test rows, with raw (untempered) logits:

- target-referenced ECE ("soft ECE"): bin rows by the model's max-prob p_max
  and compare it with the *target's* probability t[argmax p] of the option
  the model picked. A model whose probabilities equal the soft targets has
  soft ECE 0; the sign of (p_max - t[argmax p]) says over- vs under-sharp.
- the sharpness gap mean(p_max) - mean(t_max), overall, per question type,
  and per target-entropy tercile;
- hard-label ECE for comparison;
- 15-bin reliability curves (both references), for the paper's figure.

    uv run python scripts/e2/row_analysis.py --out results/e2_row_analysis.json
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
TYPES = {0: "choice", 1: "score", 2: "noul"}
N_BINS = 15


def _bins(conf):
    edges = torch.linspace(0, 1, N_BINS + 1)
    idx = torch.bucketize(conf, edges[1:-1], right=False)
    return idx


def reliability(conf: torch.Tensor, ref: torch.Tensor) -> dict:
    """15 equal-width bins on conf; per bin mean conf, mean ref, count; ECE."""
    idx = _bins(conf)
    curve, ece, n = [], 0.0, len(conf)
    for b in range(N_BINS):
        m = idx == b
        c = int(m.sum())
        if c == 0:
            continue
        mc, mr = float(conf[m].mean()), float(ref[m].float().mean())
        curve.append({"bin": b, "conf": mc, "ref": mr, "count": c})
        ece += c / n * abs(mc - mr)
    return {"ece": ece, "curve": curve}


def analyse(rows: list[dict]) -> dict:
    p = [torch.softmax(r["logits"].float(), -1) for r in rows]
    t = [r["target"].float() for r in rows]
    pmax = torch.tensor([float(x.max()) for x in p])
    top = [int(x.argmax()) for x in p]
    t_at_top = torch.tensor([float(ti[j]) for ti, j in zip(t, top)])
    tmax = torch.tensor([float(ti.max()) for ti in t])
    gold = [
        r["label"] if 0 <= r["label"] < r["k"] else int(ti.argmax())
        for r, ti in zip(rows, t)
    ]
    hit = torch.tensor([int(j == g) for j, g in zip(top, gold)])
    ent = torch.tensor([float(-(ti * ti.clamp_min(1e-12).log()).sum()) for ti in t])
    qtype = torch.tensor([r["qtype"] for r in rows])

    out = {
        "n": len(rows),
        "sharp_gap": float(pmax.mean() - tmax.mean()),
        "over_sharp_vs_pick": float((pmax - t_at_top).mean()),
        "soft": reliability(pmax, t_at_top),
        "hard": reliability(pmax, hit),
        "by_type": {},
        "by_entropy_tercile": {},
    }
    for q, name in TYPES.items():
        m = qtype == q
        if m.any():
            out["by_type"][name] = {
                "n": int(m.sum()),
                "sharp_gap": float(pmax[m].mean() - tmax[m].mean()),
                "soft_ece": reliability(pmax[m], t_at_top[m])["ece"],
            }
    qs = torch.quantile(ent, torch.tensor([1 / 3, 2 / 3]))
    for i, name in enumerate(("low", "mid", "high")):
        lo = -1 if i == 0 else float(qs[i - 1])
        hi = float("inf") if i == 2 else float(qs[i])
        m = (ent > lo) & (ent <= hi)
        out["by_entropy_tercile"][name] = {
            "n": int(m.sum()),
            "target_entropy_mean": float(ent[m].mean()),
            "sharp_gap": float(pmax[m].mean() - tmax[m].mean()),
            "soft_ece": reliability(pmax[m], t_at_top[m])["ece"],
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs_dir", default=str(REPO_ROOT / "checkpoints" / "e2"))
    parser.add_argument(
        "--out", default=str(REPO_ROOT / "results" / "e2_row_analysis.json")
    )
    args = parser.parse_args()

    per_run = {}
    teacher = None
    for d in sorted(Path(args.runs_dir).iterdir()):
        f = d / "logits.pt"
        if not f.exists():
            continue
        dump = torch.load(f, map_location="cpu", weights_only=False)
        per_run[d.name] = analyse(dump["test"])
        if teacher is None:  # the teacher as a predictor: logits = log t
            rows = [
                {**r, "logits": r["target"].clamp_min(1e-9).log()} for r in dump["test"]
            ]
            teacher = analyse(rows)

    groups = defaultdict(list)
    for name, res in per_run.items():
        m = re.match(r"(.+)_seed\d+(_.+)?$", name)
        if m and not m.group(2):
            groups[m.group(1)].append(res)

    def mean(xs):
        return sum(xs) / len(xs)

    summary = {
        cfg: {
            "n_seeds": len(rs),
            "sharp_gap": mean([r["sharp_gap"] for r in rs]),
            "soft_ece": mean([r["soft"]["ece"] for r in rs]),
            "hard_ece": mean([r["hard"]["ece"] for r in rs]),
            "by_type_sharp_gap": {
                t: mean([r["by_type"][t]["sharp_gap"] for r in rs])
                for t in rs[0]["by_type"]
            },
            "by_entropy_sharp_gap": {
                t: mean([r["by_entropy_tercile"][t]["sharp_gap"] for r in rs])
                for t in rs[0]["by_entropy_tercile"]
            },
        }
        for cfg, rs in groups.items()
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(
            {"teacher": teacher, "per_run": per_run, "per_config": summary}, indent=2
        )
    )
    print(
        f"{'config':26s} n  sharp_gap  soft_ECE  hard_ECE | gap choice/score/noul | gap by target entropy low/mid/high"
    )
    print(
        f"{'teacher (t itself)':26s} -  {teacher['sharp_gap']:+.3f}     {teacher['soft']['ece']:.3f}     {teacher['hard']['ece']:.3f}"
    )
    for cfg, s in summary.items():
        bt, be = s["by_type_sharp_gap"], s["by_entropy_sharp_gap"]
        print(
            f"{cfg:26s} {s['n_seeds']}  {s['sharp_gap']:+.3f}     {s['soft_ece']:.3f}     {s['hard_ece']:.3f} | "
            f"{bt['choice']:+.3f} {bt['score']:+.3f} {bt['noul']:+.3f} | "
            f"{be['low']:+.3f} {be['mid']:+.3f} {be['high']:+.3f}"
        )
    tt = teacher["by_entropy_tercile"]
    print(
        "target entropy tercile means (nats):",
        {k: round(v["target_entropy_mean"], 3) for k, v in tt.items()},
    )


if __name__ == "__main__":
    main()
