"""Export one finished E2 run to a Hugging Face repo.

Converts `checkpoints/<run>/epoch_<N>.pt` (weights + AdamW state, ~5 GB)
into a weights-only `model.safetensors` (~1.7 GB fp32, loadable with
`TrainingConfig.init_from` or `DecisionModel.load_state_dict`) and uploads
it with the run's results into `<repo>/<run_name>/`:

    model.safetensors   DecisionModel state_dict (incl. the temperature buffer)
    model_config.json   DecisionModelConfig the weights were trained with
    test_eval.json  train_log.json  grad_diagnostics.json  config.json

Needs HF_TOKEN in the environment. Creates a **public** repo by default so
the paper's result files are actually publicly reachable; pass `--private`
for a private one. Usage:
    uv run python scripts/e2/export_to_hf.py --run_name e2_laya_rlce_seed42 \
        --repo minhleduc/rlcd-e2-checkpoints
"""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import torch
from huggingface_hub import HfApi
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--ckpt_dir", default=str(REPO_ROOT / "checkpoints"))
    parser.add_argument("--result_dir", default=str(REPO_ROOT / "results"))
    parser.add_argument(
        "--private",
        action="store_true",
        help="create/keep the repo private (default: public, so results are reachable)",
    )
    args = parser.parse_args()

    run_ckpt = Path(args.ckpt_dir) / args.run_name
    ckpts = sorted(run_ckpt.glob("epoch_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not ckpts:
        sys.exit(f"no epoch_*.pt under {run_ckpt}")
    ckpt_path = ckpts[-1]
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        state = {k: v.contiguous() for k, v in raw["model"].items()}
        save_file(state, str(out / "model.safetensors"))
        meta = {
            **raw.get("extra", {}).get("cfg", {}),
            "source_checkpoint": ckpt_path.name,
            "epoch": raw.get("extra", {}).get("epoch"),
            "step": raw.get("step"),
        }
        (out / "model_config.json").write_text(json.dumps(meta, indent=2, default=str))
        for name in (
            "test_eval.json",
            "train_log.json",
            "grad_diagnostics.json",
            "config.json",
        ):
            src = Path(args.result_dir) / args.run_name / name
            if src.exists():
                shutil.copy(src, out / name)

        api = HfApi()
        api.create_repo(
            args.repo, repo_type="model", private=args.private, exist_ok=True
        )
        api.upload_folder(
            folder_path=str(out),
            path_in_repo=args.run_name,
            repo_id=args.repo,
            repo_type="model",
            commit_message=f"E2 run {args.run_name} ({ckpt_path.name})",
        )
    print(
        f"uploaded {args.run_name} -> https://huggingface.co/{args.repo}/tree/main/{args.run_name}"
    )


if __name__ == "__main__":
    main()
