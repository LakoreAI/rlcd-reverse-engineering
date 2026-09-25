"""Aggregate E2 runs into report tables (markdown + JSON).

Reads every `<dir>/<run_name>/{test_eval,train_log,grad_diagnostics}.json`
under `--runs_dir` (default: the HF snapshot in checkpoints/e2) and prints:
  1. one row per run: raw/post-T metrics, over-confidence gap, fitted T;
  2. per-config mean ± std over seeds (runs named ..._seed<N>[_tag]);
  3. E3 gradient diagnostics: mean cos(∇RL, ∇CE) and ‖∇RL‖/‖∇CE‖ per sigma.

    uv run python scripts/e2/summarize.py --json results/e2_summary.json
"""

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TYPES = {"0": "choice", "1": "score", "2": "noul"}


def load_run(d: Path) -> dict | None:
    # test_eval_v2.json (from dump_logits.py) has the newer metrics, e.g.
    # mean_target_max; fall back to the training-time test_eval.json.
    ev = d / "test_eval_v2.json"
    if not ev.exists():
        ev = d / "test_eval.json"
    if not ev.exists():
        return None
    run = {"name": d.name, "eval": json.loads(ev.read_text())}
    for key in ("train_log", "grad_diagnostics", "config"):
        f = d / f"{key}.json"
        run[key] = json.loads(f.read_text()) if f.exists() else None
    m = re.match(r"(.+)_seed(\d+)(?:_(.+))?$", d.name)
    run["config_name"], run["seed"], run["tag"] = (
        (m.group(1), int(m.group(2)), m.group(3)) if m else (d.name, None, None)
    )
    return run


def fitted_t(run: dict) -> dict[str, float]:
    out = {}
    for key, t in run["eval"]["fitted_temperature"].items():
        m = re.match(r"type(\d)_bucket(\d)", key)
        out[f"{TYPES[m.group(1)]}/b{m.group(2)}"] = t
    return out


def row(run: dict) -> dict:
    raw, post = run["eval"]["raw"], run["eval"]["post_temperature"]
    return {
        "run": run["name"],
        "raw_ece": raw["ece"],
        "post_ece": post["ece"],
        "brier": raw["brier"],
        "nll": raw["nll"],
        "acc": raw["accuracy"],
        "soft_acc": raw.get("soft_accuracy"),
        "score_mae": raw.get("score_mae"),
        # > 0: over-confident on average (mean max-prob above hit rate)
        "conf_minus_acc": raw["mean_confidence"] - raw["accuracy"]
        if "mean_confidence" in raw
        else None,
        # > 0: sharper than the soft targets (the §V claim's reference point)
        "conf_minus_target": raw["mean_confidence"] - raw["mean_target_max"]
        if "mean_target_max" in raw
        else None,
        **{f"T[{k}]": v for k, v in fitted_t(run).items()},
    }


def fmt(v, nd=3):
    return "–" if v is None else f"{v:.{nd}f}"


def md_table(rows: list[dict], cols: list[str]) -> str:
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                r[c] if isinstance(r.get(c), str) else fmt(r.get(c)) for c in cols
            )
            + " |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs_dir", default=str(REPO_ROOT / "checkpoints" / "e2"))
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    runs = [
        r
        for d in sorted(Path(args.runs_dir).iterdir())
        if d.is_dir() and (r := load_run(d))
    ]
    rows = [row(r) for r in runs]
    t_cols = sorted({c for r in rows for c in r if c.startswith("T[")})
    cols = [
        "run",
        "conf_minus_target",
        "raw_ece",
        "post_ece",
        "brier",
        "nll",
        "acc",
        "soft_acc",
        "score_mae",
        "conf_minus_acc",
    ] + t_cols
    print("## Per run\n")
    print(md_table(rows, cols))

    # per config (untagged runs only), mean ± std over seeds
    groups = defaultdict(list)
    for run, r in zip(runs, rows):
        if run["tag"] is None:
            groups[run["config_name"]].append(r)
    agg_rows = []
    for name, rs in groups.items():
        agg = {"config": name, "n_seeds": str(len(rs))}
        for c in [
            "conf_minus_target",
            "raw_ece",
            "post_ece",
            "brier",
            "nll",
            "acc",
            "conf_minus_acc",
        ] + t_cols:
            vals = [r[c] for r in rs if r.get(c) is not None]
            if vals:
                sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
                agg[c] = f"{statistics.mean(vals):.3f} ± {sd:.3f}"
        agg_rows.append(agg)
    print("\n## Per config (mean ± std over seeds)\n")
    print(
        md_table(
            agg_rows,
            [
                "config",
                "n_seeds",
                "conf_minus_target",
                "raw_ece",
                "post_ece",
                "brier",
                "nll",
                "acc",
                "conf_minus_acc",
            ]
            + t_cols,
        )
    )

    # E3
    grad_rows = []
    for run in runs:
        g = run["grad_diagnostics"] or []
        by_sigma = defaultdict(list)
        for e in g:
            by_sigma[round(e["sigma"], 3)].append(e)
        for sigma, es in sorted(by_sigma.items()):
            grad_rows.append(
                {
                    "run": run["name"],
                    "sigma": sigma,
                    "n": str(len(es)),
                    "cos_mean": statistics.mean(e["grad_cos_rl_ce"] for e in es),
                    "cos_std": statistics.stdev(e["grad_cos_rl_ce"] for e in es)
                    if len(es) > 1
                    else 0.0,
                    "ratio_mean": statistics.mean(
                        e["grad_norm_ratio_rl_ce"] for e in es
                    ),
                }
            )
    print("\n## E3: gradient diagnostics (w.r.t. logits)\n")
    print(
        md_table(grad_rows, ["run", "sigma", "n", "cos_mean", "cos_std", "ratio_mean"])
    )

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(
            json.dumps(
                {"runs": rows, "per_config": agg_rows, "grad": grad_rows}, indent=2
            )
        )


if __name__ == "__main__":
    main()
