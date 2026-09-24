"""Evaluate a trained checkpoint on a feature split.

Reports accuracy / macro-F1 and the per-class report, optionally writing a JSON
summary.

Usage:
    uv run python scripts/training/evaluate.py \
        --ckpt checkpoints/<run>/best.pt --data data/raw/test.npz
"""

import argparse
import sys
from pathlib import Path

from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data import dataset_from_npz  # noqa: E402
from src.pipelines.eval import evaluate, format_report  # noqa: E402
from src.pipelines.infer import load_model  # noqa: E402
from src.utils.model_utils import detect_device  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument(
        "--data", type=Path, default=REPO_ROOT / "data" / "raw" / "test.npz"
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    device = detect_device()
    model, cfg = load_model(args.ckpt, device)
    dataset = dataset_from_npz(args.data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    result = evaluate(
        model,
        loader,
        device,
        num_classes=cfg.num_classes,
        save_json_path=args.json,
    )
    print(format_report(result))

    if args.json:
        print(f"saved: {args.json}")


if __name__ == "__main__":
    main()
