"""Direct test of the noise-averaging identity from Sec. V-B.

For a model trained at fixed noise scale `sigma`, the smoothed optimum
satisfies E_eps[softmax(z* + eps)] = t (Eq. 5), while inference reports the
noise-free softmax(z*). On the test set this measures how much closer the
noise-averaged prediction is to the soft target than the noise-free one:

  - KL(t || p_noise_free) vs KL(t || E_eps p)  (lower is closer)
  - max p vs max pbar vs max t                 (sharpness)
  - "moved": fraction of rows where averaging moves p toward t

Works from a `.pt` checkpoint or the weights-only `model.safetensors` that
`scripts/e2/export_to_hf.py` publishes (with its `model_config.json`).

    uv run python scripts/e2/noise_average_probe.py \
        --ckpt <model.pt|model.safetensors> --sigma 2.0 \
        --json results/noise_avg_sigma2.json
"""

import argparse
import json
import sys
from collections import defaultdict
from functools import partial
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.config import QTYPES  # noqa: E402
from src.data import TypedDecisionDataset, collate_fn  # noqa: E402
from src.pipelines.infer import load_model  # noqa: E402
from src.utils.model_utils import detect_device  # noqa: E402

TYPE_NAME = {v: k for k, v in QTYPES.items()}


def noise_average(
    z: torch.Tensor, sigma: float, samples: int, generator
) -> torch.Tensor:
    """E_eps[softmax(z + eps)] with eps ~ N(0, sigma^2) projected to zero mean
    over the (already-valid) options of `z`."""
    eps = torch.randn(
        (samples, z.numel()), generator=generator, device=z.device, dtype=torch.float32
    )
    eps = (eps - eps.mean(dim=-1, keepdim=True)) * sigma
    return torch.softmax(z.float().unsqueeze(0) + eps, dim=-1).mean(0)


def _metrics(
    z: torch.Tensor, t: torch.Tensor, pbar: torch.Tensor, p0: torch.Tensor
) -> dict:
    kl = lambda p: float(  # noqa: E731
        (t * (t.clamp_min(1e-12).log() - p.clamp_min(1e-12).log())).sum()
    )
    a0, ab = int(p0.argmax()), int(pbar.argmax())
    return {
        "kl0": kl(p0),
        "klbar": kl(pbar),
        "sharp0": float(p0.max()),
        "sharpbar": float(pbar.max()),
        "sharpt": float(t.max()),
        "gap0": float(p0.max() - t[a0]),
        "gapbar": float(pbar.max() - t[ab]),
        "moved": float(kl(pbar) < kl(p0)),
    }


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument(
        "--sigma", type=float, required=True, help="training noise scale"
    )
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--split", default="test")
    parser.add_argument("--n_cases", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    device = detect_device()
    model, cfg, tokenizer = load_model(args.ckpt, device)
    model.eval()

    hf = load_dataset("LocalLLaMA/typed-decisions", "all", split=args.split)
    if args.n_cases is not None:
        hf = hf.select(range(min(len(hf), args.n_cases)))
    ds = TypedDecisionDataset(hf, tokenizer, cfg.max_len, cfg.head_max_len)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=partial(collate_fn, pad_token_id=tokenizer.pad_token_id),
    )
    generator = torch.Generator(device=device).manual_seed(args.seed)

    agg: dict[str, list[float]] = defaultdict(list)
    by_type: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for batch in loader:
        logits = model(
            batch["ids"].to(device),
            batch["attention_mask"].to(device),
            batch["marker_pos"].to(device),
            batch["marker_mask"].to(device),
            batch["qtype"].to(device),
        ).cpu()
        marker_mask, target, qtype = (
            batch["marker_mask"],
            batch["target"],
            batch["qtype"],
        )
        for i in range(logits.shape[0]):
            k = int(marker_mask[i].sum())
            if k < 2:
                continue
            z, t = logits[i, :k], target[i, :k]
            p0 = torch.softmax(z, dim=-1)
            pbar = noise_average(z, args.sigma, args.samples, generator)
            metrics = _metrics(z, t, pbar, p0)
            name = TYPE_NAME[int(qtype[i])]
            for key, value in metrics.items():
                agg[key].append(value)
                by_type[name][key].append(value)

    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")  # noqa: E731
    summary: dict[str, object] = {
        "ckpt": str(args.ckpt),
        "sigma": args.sigma,
        "samples": args.samples,
        "n": len(agg["kl0"]),
        **{key: mean(values) for key, values in agg.items()},
        "by_type": {
            name: {key: mean(values) for key, values in metrics.items()}
            for name, metrics in by_type.items()
        },
    }
    print(json.dumps(summary, indent=2))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
