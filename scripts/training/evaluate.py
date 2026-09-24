"""Evaluate a trained checkpoint on the typed-decisions test split.

Fits per-(type, K-bucket) temperature on a held-out calibration slice of the
train split, then reports raw and post-temperature ECE/Brier/NLL/accuracy on
the test split (docs/PLAN.md sec. 5).

Usage:
    uv run python scripts/training/evaluate.py \
        --ckpt checkpoints/<run>/best.pt --json results/eval.json
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.pipelines.config import TrainingConfig  # noqa: E402
from src.pipelines.eval import evaluate, format_report  # noqa: E402
from src.pipelines.infer import load_model  # noqa: E402
from src.pipelines.train import build_loaders  # noqa: E402
from src.utils.model_utils import detect_device  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument(
        "--dataset_name", type=str, default="LocalLLaMA/typed-decisions"
    )
    parser.add_argument("--dataset_config", type=str, default="all")
    parser.add_argument("--calib_fraction", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    device = detect_device()
    model, model_cfg, tokenizer = load_model(args.ckpt, device)

    train_cfg = TrainingConfig(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        calib_fraction=args.calib_fraction,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    _train_loader, calib_loader, test_loader, _train_dataset = build_loaders(
        train_cfg, model_cfg, tokenizer, device
    )
    if calib_loader is None:
        raise RuntimeError("--calib_fraction must be > 0 to fit temperature")

    result = evaluate(
        model, calib_loader, test_loader, device, save_json_path=args.json
    )
    print(format_report(result))

    if args.json:
        print(f"saved: {args.json}")


if __name__ == "__main__":
    main()
