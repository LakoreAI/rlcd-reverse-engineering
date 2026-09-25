"""Paired bootstrap over E2 test rows for a config-vs-config gap.

`scripts/e2/summarize.py` reports mean over seeds; this asks whether a gap as
small as CE-only's NLL edge over RL+CE (0.861 vs 0.866) survives the
item-to-item spread, by pairing per-row scores on the same test rows.

    uv run python scripts/e2/paired_nll.py \
        --runs_dir results/e2_vm/results --a e2_ce_only --b e2_laya_rlce \
        --json results/e2_paired_nll.json

Needs the `logits.pt` files (fetch with
`scripts/e2/fetch_hf.py --with_logits`).
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.pipelines.eval import paired_bootstrap_diff  # noqa: E402


def config_of(run_name: str) -> str:
    """`e2_ce_only_seed43` -> `e2_ce_only`; `..._seed42_rep2` -> `...`."""
    match = re.match(r"(.+)_seed\d+", run_name)
    return match.group(1) if match else run_name


def per_row_scores(dump: dict, metric: str) -> torch.Tensor:
    """One score per test row: NLL or Brier of the raw logits against the
    soft target. Rows are index-aligned across runs (deterministic loader)."""
    values = []
    for row in dump["test"]:
        z = row["logits"].float()
        t = row["target"].float()
        if metric == "nll":
            values.append(float(-(t * torch.log_softmax(z, dim=-1)).sum()))
        else:
            values.append(float(((torch.softmax(z, dim=-1) - t) ** 2).sum()))
    return torch.tensor(values, dtype=torch.float64)


def pooled(groups: dict[str, list[torch.Tensor]], name: str) -> torch.Tensor:
    if name not in groups:
        raise SystemExit(f"no runs for {name!r}; found {sorted(groups)}")
    return torch.stack(groups[name]).mean(dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs_dir",
        default=str(REPO_ROOT / "results" / "e2_vm" / "results"),
        help="directory of <run>/logits.pt",
    )
    parser.add_argument("--a", default="e2_ce_only")
    parser.add_argument("--b", default="e2_laya_rlce")
    parser.add_argument("--metric", choices=("nll", "brier"), default="nll")
    parser.add_argument("--n_boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    groups: dict[str, list[torch.Tensor]] = defaultdict(list)
    for run_dir in sorted(Path(args.runs_dir).iterdir()):
        logits = run_dir / "logits.pt"
        if not logits.exists():
            continue
        dump = torch.load(logits, map_location="cpu", weights_only=False)
        groups[config_of(run_dir.name)].append(per_row_scores(dump, args.metric))

    a = pooled(groups, args.a)
    b = pooled(groups, args.b)
    result = {
        "metric": args.metric,
        "a": args.a,
        "b": args.b,
        "n_a_runs": len(groups[args.a]),
        "n_b_runs": len(groups[args.b]),
        **paired_bootstrap_diff(a, b, n_boot=args.n_boot, seed=args.seed),
    }
    lo, hi, mean, p = result["lo"], result["hi"], result["mean"], result["p"]
    print(
        f"{args.metric}: {args.a} - {args.b} = {mean:+.4f} "
        f"[{lo:+.4f}, {hi:+.4f}]  p={p:.3f}  (n={result['n']} rows, "
        f"{result['n_a_runs']} vs {result['n_b_runs']} runs)"
    )
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
