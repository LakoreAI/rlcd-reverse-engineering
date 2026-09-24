"""Generate a synthetic feature dataset in the template's `.npz` format.

Useful for exercising the training/eval pipeline before wiring in real data.

Usage:
    uv run python scripts/data/make_synthetic.py \
        --out data/raw --n_samples 4000 --n_features 32 --n_classes 5
"""

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data import make_synthetic_data, save_npz  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "raw")
    parser.add_argument("--n_samples", type=int, default=4000)
    parser.add_argument("--n_features", type=int, default=32)
    parser.add_argument("--n_classes", type=int, default=5)
    parser.add_argument("--n_informative", type=int, default=10)
    parser.add_argument("--class_sep", type=float, default=1.0)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--test_fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    x, y = make_synthetic_data(
        n_samples=args.n_samples,
        n_features=args.n_features,
        n_classes=args.n_classes,
        n_informative=args.n_informative,
        class_sep=args.class_sep,
        seed=args.seed,
    )

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(x))
    x, y = x[perm], y[perm]
    n_test = int(len(x) * args.test_fraction)
    n_val = int(len(x) * args.val_fraction)
    splits = {
        "test": (x[:n_test], y[:n_test]),
        "val": (x[n_test : n_test + n_val], y[n_test : n_test + n_val]),
        "train": (x[n_test + n_val :], y[n_test + n_val :]),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    for name, (xs, ys) in splits.items():
        path = args.out / f"{name}.npz"
        save_npz(path, xs, ys)
        print(f"{path}: X{tuple(xs.shape)} y{tuple(ys.shape)}")

    print(f"classes: {args.n_classes}  input_dim: {args.n_features}")


if __name__ == "__main__":
    main()
