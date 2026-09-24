"""Single-checkpoint inference: feature vectors -> logits / probabilities.

Loads a checkpoint, rebuilds the model from its saved architecture config, and
runs a forward pass over one or all rows of an `.npz` feature file. If the file
carries labels, accuracy over the selected rows is reported too.

Usage:
    uv run python -m src.pipelines.infer \\
        --ckpt checkpoints/<run>/best.pt --features data/raw/test.npz --index 0
"""

import argparse
from dataclasses import fields
from pathlib import Path
from typing import Dict, Optional

import torch

from src.config import MLPConfig
from src.data import load_npz
from src.modules.loss import predict, probabilities
from src.modules.model import MLPClassifier
from src.utils.model_utils import detect_device


def config_from_checkpoint(ckpt: dict) -> MLPConfig:
    """Rebuild MLPConfig from a checkpoint's saved `extra["cfg"]`."""
    saved = ckpt.get("extra", {}).get("cfg", {})
    known = {f.name for f in fields(MLPConfig)}
    return MLPConfig(**{k: v for k, v in saved.items() if k in known})


def load_model(
    ckpt_path: Path, device: torch.device
) -> tuple[MLPClassifier, MLPConfig]:
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    raw = torch.load(ckpt_path, map_location=str(device))
    cfg = config_from_checkpoint(raw)
    model = MLPClassifier(cfg).to(device)
    model.load_state_dict(raw["model"])
    model.eval()
    return model, cfg


@torch.no_grad()
def infer(
    ckpt_path: Path,
    features_path: Path,
    index: Optional[int] = 0,
    out_path: Optional[Path] = None,
) -> Dict[str, object]:
    """Score one row (`index`) or every row (`index=None`) of the feature file."""
    device = detect_device()
    model, cfg = load_model(ckpt_path, device)
    x, y = load_npz(features_path)

    if index is None:
        x_sel, y_sel = x, y
    else:
        x_sel, y_sel = x[index : index + 1], y[index : index + 1]

    if x_sel.shape[1] != cfg.input_dim:
        raise ValueError(
            f"feature dim mismatch: file has {x_sel.shape[1]}, model expects {cfg.input_dim}"
        )
    x_sel = x_sel.to(device)
    logits, feature = model(x_sel)
    probs = probabilities(logits)
    preds = predict(logits)

    print(f"checkpoint: {ckpt_path}")
    print(f"samples: {x_sel.shape[0]}  feature: {tuple(feature.shape)}")
    print(f"predicted: {preds.tolist()}")
    if len(y_sel):
        matches = (preds.cpu() == y_sel).float().mean().item()
        print(f"labels:    {y_sel.tolist()}  (accuracy={matches * 100:.2f}%)")

    result = {
        "logits": logits.cpu(),
        "probabilities": probs.cpu(),
        "predictions": preds.cpu(),
        "feature": feature.cpu(),
        "source": str(features_path),
    }
    if out_path is not None:
        torch.save(result, out_path)
        print(f"saved: {out_path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument(
        "--index", type=int, default=0, help="row to score; omit with --all"
    )
    parser.add_argument("--all", action="store_true", help="score every row")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    infer(args.ckpt, args.features, None if args.all else args.index, args.out)
