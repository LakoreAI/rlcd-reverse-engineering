# Compute evaluation and rental for E2

Date: 2026-09-25. Question: can this project's training/evaluation pipeline
(ModernBERT-large, 421M params, `LocalLLaMA/typed-decisions`) run locally,
and if not, what is the cheapest way to run E2 on a budget of 50,000 VND?

## 1. Workload

| | value | source |
|---|---|---|
| train / test rows | 6,000 / 2,000 question rows (1,200 / 400 cases) | dataset |
| E2 train rows after the 10% case-level calibration split | ~5,400 | `split_train_calib` |
| sequence length, ModernBERT tokenizer | median 307, p90 ~385, max 512 (1.5% truncated) | measured |
| options per question | ≤ 5 | measured |
| model | 421.0M params (encoder + 2-layer head + scorer) | measured |

## 2. Local machine: Apple M5, 16 GB unified memory, 10-core GPU (MPS)

Full train step (forward + backward + AdamW) of `DecisionModel` on MPS with
L = 384 random tokens:

| precision | batch | s/step | rows/s | MPS driver memory |
|---|---|---|---|---|
| bf16 | 2 | 1.18 | 1.7 | 10.1 GB |
| bf16 | 4 | 1.52 | 2.6 | 11.2 GB |
| fp32 | 2 | 0.96 | 2.1 | 10.1 GB |
| fp32 | 4 | 5.23 | 0.8 | 12.2 GB (swapping) |
| fp32 | 8 | did not finish one step in >10 min | — | swap 13.8/14.3 GB |

**Verdict:** it's possible but slow. At about 2–2.6 rows/s, a 4-epoch run
(~21,600 rows) takes roughly 2.5–3 h, and only with memory-heavy apps closed.
The tests, the tiny-encoder smoke test (45 s), E1, and inference/evaluation
all run fine locally.

**Correction to the record:** my first retry of the small-batch benchmark
used `timeout`, which macOS does not have. The retries never ran, and I
wrongly reported that "batch 2 bf16 didn't finish a step". The numbers above
come from the later, real run.

## 3. Rental options on ckey.vn

ckey.vn is a GPU marketplace that bills hourly in VND from a prepaid balance.
Its own listings imply about 25,000 VND/USD. It deletes a machine's data
without notice when the balance runs out, and refunds are by request only,
within 7 days. Listings checked on 2026-09-25, ≥ 20 GB VRAM:

| GPU | VRAM | VND/h | hours for 50,000 VND | notes |
|---|---|---|---|---|
| RTX 3090 | 24 GB | ~7,600–8,000 | ~6.3–6.6 | community hosts; ~2–3× slower than A100 |
| A10 | 22.5 GB | 12,950 | ~3.9 | stable; batch 8 × accum 8 |
| RTX 4090 | 24 GB | ~16,400–21,900 | ~2.3–3 | most listings capped at 24 h |
| **A100 SXM4** | **40 GB** | **~16,000** (listing `ip-21171765`) | **~3.1** | **chosen** |
| L40S | 45 GB | 31,986+ | ~1.6 | more than needed |

T4 (15 GB, no bf16) and cards under 20 GB were ruled out.

## 4. Machine rented

`ip-21171765`: 1× NVIDIA A100-SXM4-40GB, driver 550.120 (CUDA 12.4), host
Ryzen 9 7950X, 32 visible cores, 124 GB RAM, Ubuntu 22.04.4. The container
has a **40 GB overlay disk** and **no Python or C compiler**. The only access
is a **ttyd web terminal** on port 3690, with no SSH port forwarded.

Consequences and fixes:

- The code went over as a base64 heredoc through the ttyd websocket.
  sha256 `0fb8f249…0bab` matched on both ends. The helper scripts live in
  the session scratchpad and are not part of the repo.
- The first probe failed with `RuntimeError: Failed to find C compiler`,
  because ModernBERT's `torch.compile`/Triton path needs `gcc`. Fixed with
  `apt-get install gcc libc6-dev`, about 1 minute.
- Disk: after `uv sync` and the downloads, 22 of 40 GB is used, and each
  run's final checkpoint is ~5 GB. We ran with `DELETE_CKPTS=1`; only the
  JSON results are kept.

## 5. Measured throughput on the A100

Probe: 60 cases, 1 epoch, bf16, batch 16 × accum 4. It ran at **124.5 train
rows/s**, including warm-up. That predicted about **3 min per 4-epoch run,
~1,800 VND for two runs**, far below the 35,000 VND stop-and-ask limit set
beforehand. It also means the plan's "≤ 5 h per run on a T4" budget is very
conservative for a single A100.

Actual use: 13 full runs of about 8 min each (training ~5.5 min, plus eval
and upload), the logit dump and one lost first batch, over about 2.4 h of VM
time. That came to **about 36,000 of the 50,000 VND balance**. Results:
`docs/reports/2026-09-25/e2_minimal_runs.md`.
