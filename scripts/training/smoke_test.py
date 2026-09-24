"""End-to-end smoke test against the real typed-decisions dataset with a
tiny encoder (see `configs/rlcd_smoke.yaml`).

Runs one epoch, evaluates the best checkpoint, and checks that the full
pipeline produces finite raw and post-temperature calibration metrics.
Needs network access (Hugging Face Hub) but no GPU. Not meant to produce
meaningful calibration numbers — see docs/PLAN.md sec. 5 for the real E2
configs, which need a T4-class GPU and ModernBERT-large.

Usage:
    uv run python scripts/training/smoke_test.py
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.pipelines.config import load_training_config  # noqa: E402
from src.pipelines.train import train  # noqa: E402


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="rlcd_smoke_"))
    try:
        cfg = load_training_config(
            REPO_ROOT / "configs" / "rlcd_smoke.yaml",
            ckpt_dir=str(tmp / "checkpoints"),
            result_dir=str(tmp / "results"),
        )
        _history, final_result = train(cfg)

        assert final_result is not None, (
            "no final evaluation produced (calib_fraction <= 0?)"
        )
        assert np.isfinite(final_result["raw"]["ece"]), "raw ECE is not finite"
        assert np.isfinite(final_result["post_temperature"]["ece"]), (
            "post-temperature ECE is not finite"
        )
        print("\nsmoke test: PASS")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
