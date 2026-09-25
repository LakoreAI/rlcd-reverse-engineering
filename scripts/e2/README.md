# E2 on a rented GPU (minimum budget)

Runs the two E2 configs that test prediction 1 in `docs/PLAN.md` §5 —
**CE-only raw ECE ≤ RL+CE raw ECE** — on one rented CUDA GPU, logging E3's
gradient diagnostics along the way. Nothing here needs editing on the
remote machine.

| Config | What it is |
|---|---|
| `configs/e2/laya_rlce.yaml` | RL+CE, a single-GPU replica of Laya's own typed-decisions fine-tune (starts from `convaiinnovations/laya`) |
| `configs/e2/ce_only.yaml` | the same, with `w_rl: 0` |
| `configs/e2/rlce_sigma1_fixed.yaml` | optional third run: RL+CE with σ held at 1.0 (prediction 2) |

Both runs use seed 42 and a fixed budget of 4 epochs, then evaluate the
final weights. There's no early stopping and no checkpoint picked by ECE.

## Machine to rent (ckey.vn, prices as of 2026-09-25)

1. **1× A100 SXM4 40 GB, US (listing ip-21171765): ~16,000 VND/h** (99.5% uptime, 167 h max, community host). This is
   the first choice: native bf16, room for Laya's batch, and datacenter hosting.
2. 1× A10 22.5 GB, NL/US: 12,950 VND/h (99.9% uptime). Use `BATCH=8 ACCUM=8`.
3. 1× RTX 3090 24 GB, CZ: ~7,975 VND/h (98.9%). This is a community host and
   about 2–3× slower than the A100.

Avoid T4 (no bf16, 15 GB) and cards under 20 GB. Avoid listings with low
uptime or a 24 h cap.

## Steps

```bash
# 0. on ckey.vn: rent the machine, wait for "Online", copy host/port/password
#    from GPU Manager.

# 1. from this repo on your laptop (asks for the password once)
bash scripts/e2/push.sh <host> <port>

# 2. on the machine
ssh root@<host> -p <port>
cd ~/rlcd-reverse-engineering
nohup bash scripts/e2/run_min.sh > e2.out 2>&1 &
tail -f e2.out        # Ctrl-C only stops tail; the run keeps going
```

About 2 minutes in, the probe prints an estimate such as
`~12 min per run … ≈ 7,000 VND`. Check it against your balance. To add the
third run when there's room, wait for the first two to finish and then run
`CONFIGS=rlce_sigma1_fixed nohup bash scripts/e2/run_min.sh > e2b.out 2>&1 &`.
Setup and downloads are skipped quickly the second time.

```bash
# 3. back on the laptop, BEFORE the balance runs out
bash scripts/e2/fetch.sh <host> <port>
# 4. stop / delete the rental on ckey.vn right away (billing is hourly)
```

## If something goes wrong

| Symptom | Fix |
|---|---|
| `CUDA out of memory` in the probe | `BATCH=8 ACCUM=8 bash scripts/e2/run_min.sh` (same effective batch); still OOM → add `GRAD_CKPT=1` |
| `torch sees no CUDA device` | driver too old for the cu126 torch wheel, so rent a different machine |
| SSH dropped | the run continues under `nohup`; reconnect and `tail -f e2.out` |
| Balance about to run out | run `fetch.sh` first. CKEY deletes stopped machines' data without notice. |

## What comes back

For each run, `results/<run_name>/` contains:
- `test_eval.json`: raw and post-temperature ECE, Brier, NLL, accuracy against the gold
  label (as Laya's eval does), soft accuracy, score MAE, the same split by question type,
  and fitted T per (type, K-bucket)
- `train_log.json`: per-epoch loss, σ, time, and calibration-slice raw metrics
- `grad_diagnostics.json`: E3's cos(∇RL, ∇CE) and ‖∇RL‖/‖∇CE‖, taken w.r.t. the logits
- `config.json`: the exact model and training config

`results/e2_logs/` also holds the full stdout of each run, the GPU and driver
info, `pip freeze`, and `GIT_REV.txt`.

## Reproducing the paper without a GPU

The weights, per-run JSONs and `logits.pt` are mirrored to a Hugging Face
artifact repo by `export_to_hf.py` (public by default; pass `--private` to
keep one private). On any machine:

```bash
uv run python scripts/e2/fetch_hf.py --repo minhleduc/rlcd-e2-checkpoints --with_logits
uv run python scripts/e2/summarize.py --runs_dir results/e2_vm/results --json results/e2_summary.json
```

That is exactly what `docs/paper/README.md` assumes, so the paper's tables and
figures regenerate without ever renting a GPU.

## Plan for the next rented run (Runpod)

To finish §9.4 of `docs/TODO.md` — 2 extra seeds each for σ=0.5, σ=2 and
RL-only (6 runs), the 3 E4 reward-composition runs, and E5 (inference-only)
— on Runpod instead of ckey.vn.

**GPU (Community Cloud is the economy tier).** Prices scraped from
runpod.io/pricing on 2026-09-13; re-check before renting.

| GPU | VRAM | bf16 | $/hr | note |
|---|---|---|---|---|
| RTX A5000 | 24 GB | yes | **0.16** | cheapest viable; only 25 GB host RAM |
| RTX 3090 | 24 GB | yes | **0.22** | best all-round (125 GB RAM, 16 vCPU) |
| RTX A6000 | 48 GB | yes | 0.33 | zero-fuss headroom |
| A40 | 48 GB | yes | 0.35 | zero-fuss headroom |
| RTX 4090 | 24 GB | yes | 0.34 | fastest 24 GB |
| A100 80 GB | 80 GB | yes | 1.19 | overkill, skip |

Pick **RTX 3090 (Community)**; fall back to A5000 for the cheapest, or
A6000/A40 48 GB if OOM. Avoid V100 (no bf16) and <24 GB cards.

**Storage.** `40 GB` container disk is ample: ~7 GB `.venv` (torch cu126 +
`nvidia/*`), ~3.4 GB HF weights (ModernBERT-large + Laya), ~5 GB OS/tools,
and ~5 GB per `epoch_*.pt` if `DELETE_CKPTS=1` keeps only the current run.
Container disk is $0.10/GB/mo (~$0.005/hr) and **ephemeral** — upload to the
HF repo before terminating. Without `DELETE_CKPTS=1`, 9 runs' checkpoints
need ~45 GB, so budget 60–70 GB instead. Attach a network volume only if
runs will span sessions.

**Deploy.**

- Template: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
  (ships `gcc`, needed for ModernBERT's Triton path).
- Env: `HF_TOKEN`, `WANDB_API_KEY`.
- 24 GB cards: `BATCH=8 ACCUM=8` (same effective batch); add `GRAD_CKPT=1`
  only if it still OOMs.

```bash
CONFIGS="rlce_sigma0p5_fixed:43 rlce_sigma0p5_fixed:44 rlce_sigma2_fixed:43 rlce_sigma2_fixed:44 rl_only:43 rl_only:44" \
HF_REPO=<user>/rlcd-e2-checkpoints DELETE_CKPTS=1 BATCH=8 ACCUM=8 \
bash scripts/e2/run_min.sh
# then the E4 runs with CONFIGS pointing at configs/e4/*.yaml
```

**Cost.** ~2–3 GPU-h total → **~$0.50–$1.20** at $0.22–0.34/hr, versus ~$4+
on an A100. Terminate the pod when done (per-second billing; stopped pods
still accrue disk). `run_min.sh` prints its estimate in VND via
`VND_PER_HOUR`; set it to the USD rate × the VND/USD rate just for the cost
line, or ignore it.
