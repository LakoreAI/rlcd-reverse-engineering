"""End-to-end smoke test on synthetic features.

Generates a small synthetic dataset, runs a couple of training epochs,
evaluates the best checkpoint, and checks that the full pipeline produces a
finite accuracy. No external dataset required.

Usage:
    uv run python scripts/training/smoke_test.py
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data import dataset_from_npz, make_synthetic_data, save_npz  # noqa: E402
from src.pipelines.config import TrainingConfig  # noqa: E402
from src.pipelines.eval import evaluate, format_report  # noqa: E402
from src.pipelines.infer import load_model  # noqa: E402
from src.pipelines.train import train  # noqa: E402
from src.utils.model_utils import detect_device  # noqa: E402

N_TRAIN, N_TEST = 600, 200
N_FEATURES, N_CLASSES = 16, 4


def build_data(root: Path) -> None:
    x, y = make_synthetic_data(
        n_samples=N_TRAIN + N_TEST,
        n_features=N_FEATURES,
        n_classes=N_CLASSES,
        n_informative=8,
        class_sep=1.5,
        seed=0,
    )
    save_npz(root / "train.npz", x[:N_TRAIN], y[:N_TRAIN])
    save_npz(root / "test.npz", x[N_TRAIN:], y[N_TRAIN:])


def main() -> None:
    torch.manual_seed(0)
    tmp = Path(tempfile.mkdtemp(prefix="mlp_smoke_"))
    try:
        build_data(tmp)
        cfg = TrainingConfig(
            data_root=str(tmp),
            train_file="train.npz",
            val_file=None,
            test_file="test.npz",
            ckpt_dir=str(tmp / "checkpoints"),
            result_dir=str(tmp / "results"),
            epochs=2,
            batch_size=32,
            eval_every=1,
            ckpt_every=1,
            log_every=5,
            lr=1e-3,
            lr_scheduler={"type": "none"},
            save_best=True,
            best_metric="accuracy",
            best_mode="max",
        )
        train(cfg)

        best = list((tmp / "checkpoints").glob("*/best.pt"))
        if not best:
            raise RuntimeError("no best.pt produced")
        device = detect_device()
        model, model_cfg = load_model(best[0], device)
        dataset = dataset_from_npz(tmp / "test.npz")
        loader = DataLoader(dataset, batch_size=64, shuffle=False)
        result = evaluate(model, loader, device, num_classes=model_cfg.num_classes)
        print(format_report(result))
        assert np.isfinite(result["accuracy"]), "accuracy is not finite"
        print("\nsmoke test: PASS")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
