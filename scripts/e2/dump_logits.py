"""Re-run finished E2 models over the calibration and test splits and save
their raw logits, so any later metric can be computed offline without a GPU.

For each `<runs_dir>/<run_name>/model.safetensors` (the layout of the HF
snapshot, see export_to_hf.py) writes `<runs_dir>/<run_name>/logits.pt`:

    {"calib": rows, "test": rows}   rows = list of {"logits", "target",
                                     "qtype", "k", "label"} (unpadded)

and `test_eval_v2.json` (the current `evaluate()` metrics, recomputed from
those rows). The calibration split is rebuilt from the run's own saved
TrainingConfig (dataset, calib_fraction, seed, max_examples), so it is the
exact slice the run held out.

    uv run python scripts/e2/dump_logits.py --runs_dir checkpoints/e2 [--upload_repo R]
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.config import DecisionModelConfig  # noqa: E402
from src.modules.model import DecisionModel  # noqa: E402
from src.pipelines.config import TrainingConfig  # noqa: E402
from src.pipelines.eval import (  # noqa: E402
    _metrics_with_breakdown,
    _scale_rows,
    collect_rows,
    fit_temperatures,
)
from src.pipelines.train import build_loaders  # noqa: E402
from src.utils.model_utils import detect_device  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs_dir", default=str(REPO_ROOT / "checkpoints" / "e2"))
    parser.add_argument("--only", nargs="*", default=None, help="run names")
    parser.add_argument("--max_examples", type=int, default=None, help="smoke test")
    parser.add_argument("--upload_repo", default=None)
    args = parser.parse_args()

    device = detect_device()
    run_dirs = [
        d
        for d in sorted(Path(args.runs_dir).iterdir())
        if (d / "model.safetensors").exists()
        and (args.only is None or d.name in args.only)
    ]
    for d in run_dirs:
        saved = json.loads((d / "config.json").read_text())
        model_cfg = DecisionModelConfig(**saved["model"])
        tr = saved["training"]
        train_cfg = TrainingConfig(
            dataset_name=tr["dataset_name"],
            dataset_config=tr["dataset_config"],
            calib_fraction=tr["calib_fraction"],
            seed=tr["seed"],
            max_examples=args.max_examples or tr.get("max_examples"),
            batch_size=32,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_cfg.encoder_name)
        model = DecisionModel(model_cfg)
        model.load_state_dict(load_file(str(d / "model.safetensors")))
        model.to(device).eval()
        _, calib_loader, test_loader, _ = build_loaders(
            train_cfg, model_cfg, tokenizer, device
        )
        with torch.autocast(
            device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            calib_rows = collect_rows(model, calib_loader, device)
            test_rows = collect_rows(model, test_loader, device)
        for rows in (calib_rows, test_rows):
            for r in rows:
                r["logits"] = r["logits"].float()
        torch.save({"calib": calib_rows, "test": test_rows}, d / "logits.pt")

        boundaries = tuple(model_cfg.k_buckets)
        temps = fit_temperatures(calib_rows, boundaries)
        result = {
            "raw": _metrics_with_breakdown(test_rows),
            "post_temperature": _metrics_with_breakdown(
                _scale_rows(test_rows, temps, boundaries)
            ),
            "fitted_temperature": {
                f"type{k[0]}_bucket{k[1]}": v for k, v in temps.items()
            },
        }
        (d / "test_eval_v2.json").write_text(json.dumps(result, indent=2))
        print(
            f"{d.name}: test n={len(test_rows)} raw ECE={result['raw']['ece']:.4f} "
            f"conf={result['raw']['mean_confidence']:.3f} target_max={result['raw']['mean_target_max']:.3f}",
            flush=True,
        )
        if args.upload_repo:
            from huggingface_hub import HfApi

            api = HfApi()
            for name in ("logits.pt", "test_eval_v2.json"):
                api.upload_file(
                    path_or_fileobj=str(d / name),
                    path_in_repo=f"{d.name}/{name}",
                    repo_id=args.upload_repo,
                    commit_message=f"{d.name}: {name}",
                )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
