"""Download the E2 result files from the public HF artifact repo.

The paper claims every table and figure regenerates from JSON result files,
but `results/` is git-ignored and the E2 runs happened on a rented machine.
This pulls the small per-run files (and optionally the ~MB `logits.pt`) into
`results/e2_vm/results/<run>/`, which is the `--runs_dir` that
`scripts/e2/summarize.py`, `docs/paper/tables/make_tables.py` and
`docs/paper/figures/make_figures.py` expect.

    uv run python scripts/e2/fetch_hf.py --repo minhleduc/rlcd-e2-checkpoints
    uv run python scripts/e2/fetch_hf.py --repo <repo> --with_logits

Needs no token if the repo is public (see export_to_hf.py).
"""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ROOT = Path(__file__).resolve().parents[2]

SMALL_FILES = (
    "test_eval_v2.json",
    "test_eval.json",
    "train_log.json",
    "grad_diagnostics.json",
    "config.json",
    "model_config.json",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="e.g. <user>/rlcd-e2-checkpoints")
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "results" / "e2_vm" / "results"),
        help="local runs directory (default: results/e2_vm/results)",
    )
    parser.add_argument(
        "--with_logits",
        action="store_true",
        help="also fetch each run's logits.pt (needed by row_analysis.py / the paper figures)",
    )
    parser.add_argument("--revision", default=None, help="optional repo revision")
    args = parser.parse_args()

    files = list(SMALL_FILES) + (["logits.pt"] if args.with_logits else [])
    # snapshot_download keeps the repo layout, and `<run>/<file>` globs map to
    # local_dir/<run>/<file>.
    allow_patterns = [f"*/{name}" for name in files]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=args.repo,
        repo_type="model",
        revision=args.revision,
        allow_patterns=allow_patterns,
        local_dir=str(out),
    )

    runs = sorted(d.name for d in out.iterdir() if d.is_dir())
    print(f"fetched {args.repo} -> {path}")
    print(f"{len(runs)} run(s): {', '.join(runs) if runs else '(none)'}")


if __name__ == "__main__":
    main()
