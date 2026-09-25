#!/usr/bin/env bash
# Minimal-budget E2 on ONE rented CUDA GPU: environment setup, a short
# speed/OOM probe with a time + VND estimate, then the E2 configs one after
# another, each logged and its JSON results kept under results/.
#
# Run it detached so an SSH drop doesn't kill it:
#   cd ~/rlcd-reverse-engineering
#   nohup bash scripts/e2/run_min.sh > e2.out 2>&1 &
#   tail -f e2.out
#
# Environment knobs:
#   CONFIGS        runs to do, in order: "<config>[:<seed>[:<tag>]] ...", where
#                  <config> is configs/e2/<config>.yaml. A seed overrides the
#                  config's and renames the run ..._seed<seed>; a tag is appended
#                  (e.g. "laya_rlce:42:rep2" repeats a run under a new name).
#                  Default "laya_rlce ce_only".
#   SKIP_PROBE     1 = skip the probe (machine already measured)
#   BATCH / ACCUM  micro-batch and accumulation (default 16 / 4 = effective 64,
#                  Laya's). On OOM use BATCH=8 ACCUM=8 (same effective batch).
#   GRAD_CKPT      1 = gradient checkpointing (less memory, ~30% slower)
#   VND_PER_HOUR   the machine's price, for the cost estimate (default: A100 listing)
#   PROBE_ONLY     1 = stop after the probe and estimate
#   DELETE_CKPTS   1 = delete checkpoints/ after each run (small disks)
#   HF_REPO        e.g. minhleduc/rlcd-e2-checkpoints: after each run, upload
#                  weights-only model.safetensors + results there (needs HF_TOKEN)
#   HF_TOKEN / WANDB_API_KEY   read from the environment or from ./.env
set -euo pipefail
cd "$(dirname "$0")/../.."

CONFIGS=${CONFIGS:-"laya_rlce ce_only"}
BATCH=${BATCH:-16}
ACCUM=${ACCUM:-4}
GRAD_CKPT=${GRAD_CKPT:-0}
VND_PER_HOUR=${VND_PER_HOUR:-16000}
LOG_DIR=results/e2_logs
mkdir -p "$LOG_DIR"
T0=$(date +%s)
stamp() { echo "[$(date '+%H:%M:%S') +$(( ($(date +%s) - T0) / 60 ))min] $*"; }

COMMON=(--batch_size "$BATCH" --accum_steps "$ACCUM")
[ "$GRAD_CKPT" = "1" ] && COMMON+=(--gradient_checkpointing true)

# --- 1. environment -----------------------------------------------------
stamp "setup"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
if ! command -v gcc >/dev/null && command -v apt-get >/dev/null; then
  # ModernBERT's torch.compile/Triton path needs a C compiler
  DEBIAN_FRONTEND=noninteractive apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq gcc libc6-dev >/dev/null
fi
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh || pip install -q uv
  export PATH="$HOME/.local/bin:$PATH"
fi
uv sync --extra rich --extra wandb
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | tee "$LOG_DIR/gpu.txt"
uv run python - <<'PY' | tee -a "$LOG_DIR/gpu.txt"
import torch
assert torch.cuda.is_available(), (
    "torch sees no CUDA device - the machine's driver may be too old for the "
    "cu126 torch wheel; pick another machine"
)
p = torch.cuda.get_device_properties(0)
print(f"torch {torch.__version__} | {p.name} | {p.total_memory / 2**30:.1f} GB | "
      f"bf16={torch.cuda.is_bf16_supported()}")
assert torch.cuda.is_bf16_supported(), "no bf16 on this GPU - set amp_dtype: fp16"
PY
uv pip freeze > "$LOG_DIR/pip_freeze.txt"
[ -f GIT_REV.txt ] && cp GIT_REV.txt "$LOG_DIR/"

# --- 2. downloads (once, so the runs don't pay for them) -------------------
stamp "downloading dataset + ModernBERT-large + Laya weights"
uv run python - <<'PY'
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer
load_dataset("LocalLLaMA/typed-decisions", "all")
AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large")
AutoModel.from_pretrained("answerdotai/ModernBERT-large")
hf_hub_download("convaiinnovations/laya", "model.safetensors")
PY

# --- 3. probe: catches OOM / errors in ~2 min and times a real epoch slice --
if [ "${SKIP_PROBE:-0}" != "1" ]; then
stamp "probe (60 cases, 1 epoch)"
rm -rf /tmp/e2_probe /tmp/e2_probe_ckpt
uv run python -m src.pipelines.train --config configs/e2/laya_rlce.yaml \
  --max_examples 60 --epochs 1 --run_name _probe \
  --ckpt_dir /tmp/e2_probe_ckpt --result_dir /tmp/e2_probe "${COMMON[@]}" \
  2>&1 | tee "$LOG_DIR/probe.log"
N_CONFIGS=$(echo "$CONFIGS" | wc -w | tr -d ' ')
VND_PER_HOUR=$VND_PER_HOUR N_CONFIGS=$N_CONFIGS uv run python - <<'PY' | tee "$LOG_DIR/estimate.txt"
import json, os
rec = json.load(open("/tmp/e2_probe/_probe/train_log.json"))[0]
sec_per_row = rec["epoch_seconds"] / rec["train_rows"]
full_rows, epochs = 5400, 4          # 1080 train cases x 5 questions
eval_overhead = 1.15                 # per-epoch calib eval + final test eval
run_min = sec_per_row * full_rows * epochs * eval_overhead / 60
n = int(os.environ["N_CONFIGS"]); vnd_h = float(os.environ["VND_PER_HOUR"])
print(f"probe: {1 / sec_per_row:.1f} train rows/s (first epoch incl. warm-up, so pessimistic)")
print(f"estimate: ~{run_min:.0f} min per run, ~{n * run_min:.0f} min for {n} run(s) "
      f"~= {n * run_min / 60 * vnd_h:,.0f} VND at {vnd_h:,.0f} VND/h (plus setup time)")
PY
fi
if [ "${PROBE_ONLY:-0}" = "1" ]; then stamp "PROBE_ONLY=1 - stopping"; exit 0; fi

# --- 4. the E2 runs -----------------------------------------------------
for entry in $CONFIGS; do
  IFS=: read -r name seed tag <<< "$entry"
  run_name=$(grep -E '^run_name:' "configs/e2/$name.yaml" | awk '{print $2}')
  RUN_ARGS=()
  if [ -n "${seed:-}" ]; then
    run_name="${run_name%_seed*}_seed$seed"
    RUN_ARGS+=(--seed "$seed")
  fi
  [ -n "${tag:-}" ] && run_name="${run_name}_$tag"
  RUN_ARGS+=(--run_name "$run_name")
  stamp "run $run_name"
  uv run python -m src.pipelines.train --config "configs/e2/$name.yaml" \
    "${COMMON[@]}" "${RUN_ARGS[@]}" 2>&1 | tee "$LOG_DIR/$run_name.log"
  # results/<run_name>/ now holds test_eval.json, train_log.json,
  # grad_diagnostics.json, config.json — small; fetch.sh pulls them.
  if [ -n "${HF_REPO:-}" ]; then
    stamp "uploading $run_name to $HF_REPO"
    uv run python scripts/e2/export_to_hf.py --run_name "$run_name" --repo "$HF_REPO" \
      2>&1 | tee -a "$LOG_DIR/$run_name.log"
  fi
  if [ "${DELETE_CKPTS:-0}" = "1" ]; then
    rm -rf checkpoints/*  # ~5 GB/run; for small-disk rentals
  fi
  df -h . | tail -1
  stamp "done $run_name"
done

stamp "all runs finished - fetch results now, then STOP the machine"
