"""E5 — consistency probes for a trained typed-decision checkpoint.

Inference-only, no training. Scores each test question under controlled
perturbations and reports a violation rate:

  - permutation: reverse the option order. A consistent model picks the same
    *option content* (for score-type, the same ordinal level after relabeling).
  - renaming: append a neutral "(restated)" to every option description. The
    picked option key should not change.
  - irrelevant context: append an unrelated passage to the state. The picked
    option key should not change.
  - negation (noul only): negate the question and swap the true/false wording;
    a consistent model should mirror the distribution, i.e.
    p(true) on the original ~ p(false) on the negation. Reported as the mean
    absolute mismatch, not a 0/1 violation.

    uv run python scripts/e5/consistency_probes.py \
        --ckpt checkpoints/<run>/best.pt --json results/e5_probes.json

Needs the `LocalLLaMA/typed-decisions` test split (network) and a checkpoint.
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.config import QTYPES  # noqa: E402
from src.data import build_sequence, option_keys  # noqa: E402
from src.modules.loss import probabilities  # noqa: E402
from src.pipelines.infer import load_model  # noqa: E402
from src.utils.model_utils import detect_device  # noqa: E402

DISTRACTOR = (
    " Unrelated note: the weather in Lisbon was mild that week, and a bakery "
    "on the corner sold out of bread before noon."
)


@torch.no_grad()
def score(model, tokenizer, cfg, state, question, device) -> dict[str, float]:
    """Raw `softmax(z)` over the question's options (no temperature: it is
    monotone, so it cannot change an argmax consistency verdict)."""
    ids, markers = build_sequence(
        tokenizer, state, question, cfg.max_len, cfg.head_max_len
    )
    ids_t = torch.tensor([ids], dtype=torch.long, device=device)
    marker_pos = torch.tensor([markers], dtype=torch.long, device=device)
    qtype = torch.tensor([QTYPES[question["type"]]], dtype=torch.long, device=device)
    logits = model(
        ids_t,
        torch.ones_like(ids_t),
        marker_pos,
        torch.ones_like(marker_pos, dtype=torch.bool),
        qtype,
    )
    probs = probabilities(logits)[0].tolist()
    return dict(zip(option_keys(question), probs))


def permute_question(question: dict) -> dict:
    """Deep copy with the option order reversed (score criteria are a list,
    choice/noul criteria are keyed dicts)."""
    q = copy.deepcopy(question)
    if q["type"] == "score":
        q["criteria"] = list(reversed(q["criteria"]))
    else:
        q["criteria"] = dict(reversed(list(q["criteria"].items())))
    return q


def rename_question(question: dict) -> dict:
    """Deep copy with a neutral suffix on every option description."""
    q = copy.deepcopy(question)
    if q["type"] == "score":
        q["criteria"] = [f"{c} (restated)" for c in q["criteria"]]
    else:
        q["criteria"] = {k: f"{v} (restated)" for k, v in q["criteria"].items()}
    return q


def negate_noul(question: dict) -> dict:
    """Deep copy of a noul question with the instruction negated and the
    option wording swapped, so the answer should mirror."""
    q = copy.deepcopy(question)
    q["instructions"] = f"It is NOT the case that: {question['instructions']}"
    criteria = question.get("criteria")
    if criteria:
        q["criteria"] = {k: v for k, v in reversed(list(criteria.items()))}
    return q


def _argmax(probs: dict[str, float]) -> str:
    return max(probs, key=probs.get)


def _index(keys: list[str], key: str) -> int:
    return keys.index(key)


def probe_row(model, tokenizer, cfg, state, question, device) -> dict:
    base = score(model, tokenizer, cfg, state, question, device)
    keys = list(base)
    pred = _argmax(base)

    perm = score(model, tokenizer, cfg, state, permute_question(question), device)
    if question["type"] == "score":
        # reversing the levels relabels each option i as (k-1-i)
        perm_ok = _index(keys, _argmax(perm)) == len(keys) - 1 - _index(keys, pred)
    else:
        perm_ok = _argmax(perm) == pred

    renamed = score(model, tokenizer, cfg, state, rename_question(question), device)
    rename_ok = _argmax(renamed) == pred

    distracted = score(model, tokenizer, cfg, state + DISTRACTOR, question, device)
    context_ok = _argmax(distracted) == pred

    row = {
        "type": question["type"],
        "permutation_ok": perm_ok,
        "renaming_ok": rename_ok,
        "irrelevant_context_ok": context_ok,
    }
    if question["type"] == "noul":
        neg = score(model, tokenizer, cfg, state, negate_noul(question), device)
        # keys are ["false", "true"]; mirror: p_orig(true) vs p_neg(false)
        true_key = "true" if "true" in base else keys[-1]
        false_key = "false" if "false" in base else keys[0]
        row["negation_abs_gap"] = abs(base[true_key] - neg[false_key])
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--n_cases", type=int, default=100)
    parser.add_argument("--split", default="test")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    device = detect_device()
    model, cfg, tokenizer = load_model(args.ckpt, device)
    dataset = load_dataset("LocalLLaMA/typed-decisions", "all", split=args.split)
    dataset = dataset.select(range(min(len(dataset), args.n_cases)))

    rows = []
    for case in dataset:
        state = case["state"]
        questions = json.loads(case["questions"])
        for question in questions.values():
            rows.append(probe_row(model, tokenizer, cfg, state, question, device))

    n = len(rows)
    summary: dict[str, object] = {
        "n": n,
        "permutation_violation_rate": 1 - sum(r["permutation_ok"] for r in rows) / n,
        "renaming_violation_rate": 1 - sum(r["renaming_ok"] for r in rows) / n,
        "irrelevant_context_violation_rate": 1
        - sum(r["irrelevant_context_ok"] for r in rows) / n,
        "by_type": {},
    }
    for qtype in sorted({r["type"] for r in rows}):
        sub = [r for r in rows if r["type"] == qtype]
        entry = {
            "n": len(sub),
            "permutation_violation_rate": 1
            - sum(r["permutation_ok"] for r in sub) / len(sub),
            "renaming_violation_rate": 1
            - sum(r["renaming_ok"] for r in sub) / len(sub),
            "irrelevant_context_violation_rate": 1
            - sum(r["irrelevant_context_ok"] for r in sub) / len(sub),
        }
        gaps = [r["negation_abs_gap"] for r in sub if "negation_abs_gap" in r]
        if gaps:
            entry["negation_abs_gap_mean"] = sum(gaps) / len(gaps)
        summary["by_type"][qtype] = entry

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
