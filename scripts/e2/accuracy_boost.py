"""Inference-only accuracy boosters for the typed-decision models.

Two strategies that need no retraining:

  1. Seed/ensemble averaging: average the predicted distributions of several
     checkpoints (seeds or loss variants) over the same test rows.
  2. Option-order TTA: rerun each choice question with its option order
     reversed and average the two views (the reversed probabilities are
     flipped back, so the hard-label index is unchanged). Tests whether the
     marker readout is sensitive to option position.

Reports accuracy for each single model, the ensemble, each model + TTA, and
the ensemble + TTA, plus soft accuracy. Weights come from the public HF repo
(or local paths).

    uv run python scripts/e2/accuracy_boost.py \
        --runs e2_ce_only_seed42 e2_ce_only_seed43 e2_ce_only_seed44 \
        --tta --json results/accuracy_boost.json
"""

import argparse
import copy
import json
import sys
from functools import partial
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.config import QTYPES  # noqa: E402
from src.data import (  # noqa: E402
    build_sequence,
    collate_fn,
    label_index,
    target_vector,
)
from src.pipelines.infer import load_model  # noqa: E402
from src.utils.model_utils import detect_device  # noqa: E402


def reverse_choice(question: dict) -> dict:
    """Reverse a choice question's options; leave other types untouched."""
    if question["type"] != "choice":
        return question
    q = copy.deepcopy(question)
    q["criteria"] = dict(reversed(list(q["criteria"].items())))
    return q


class ViewDataset(Dataset):
    """One (case, question) per row, optionally with choice order reversed.

    Carries the hard-label index in the view's own option order and a
    `reversed` flag so probabilities can be flipped back.
    """

    def __init__(self, hf, tokenizer, max_len, head_max_len, reverse=False):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.head_max_len = head_max_len
        self.rows = []
        for example in hf:
            state = example["state"]
            questions = json.loads(example["questions"])
            gold = json.loads(example["gold"])
            for qid, question in questions.items():
                q = reverse_choice(question) if reverse else question
                is_reversed = reverse and question["type"] == "choice"
                self.rows.append((state, q, gold[qid], is_reversed))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        state, question, gold, is_reversed = self.rows[index]
        ids, markers = build_sequence(
            self.tokenizer, state, question, self.max_len, self.head_max_len
        )
        return {
            "ids": ids,
            "markers": markers,
            "target": target_vector(question, gold),
            "qtype": QTYPES[question["type"]],
            "label": label_index(question, gold),
            "reversed": is_reversed,
        }


def _collate(batch, pad_token_id):
    out = collate_fn(batch, pad_token_id=pad_token_id)
    out["reversed"] = torch.tensor([row["reversed"] for row in batch], dtype=torch.bool)
    return out


@torch.no_grad()
def predict_probs(model, dataset, device, batch_size):
    """Returns (list of per-row prob tensors in view order, labels, reversed)."""
    tokenizer = dataset.tokenizer
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(_collate, pad_token_id=tokenizer.pad_token_id),
    )
    probs, labels, reversed_flags = [], [], []
    model.eval()
    for batch in loader:
        logits = model(
            batch["ids"].to(device),
            batch["attention_mask"].to(device),
            batch["marker_pos"].to(device),
            batch["marker_mask"].to(device),
            batch["qtype"].to(device),
        ).cpu()
        mask = batch["marker_mask"]
        for i in range(logits.shape[0]):
            k = int(mask[i].sum())
            probs.append(torch.softmax(logits[i, :k], dim=-1))
            labels.append(int(batch["label"][i]))
            reversed_flags.append(bool(batch["reversed"][i]))
    return probs, labels, reversed_flags


def unflip(probs, flags):
    """Flip the probability vector back for rows whose options were reversed."""
    return [p.flip(0) if f else p for p, f in zip(probs, flags)]


def accuracy(probs, labels) -> float:
    correct = 0
    for p, label in zip(probs, labels):
        if 0 <= label < p.numel() and int(p.argmax()) == label:
            correct += 1
    return correct / len(labels) if labels else float("nan")


def mean_probs(list_of_prob_lists):
    """Elementwise mean across models (rows must be aligned)."""
    out = []
    for row in zip(*list_of_prob_lists):
        out.append(torch.stack(row).mean(0))
    return out


def resolve_weights(run, repo, cache):
    from huggingface_hub import snapshot_download

    dest = Path(cache) / run
    if not (dest / "model.safetensors").exists():
        snapshot_download(repo, allow_patterns=[f"{run}/*"], local_dir=cache)
    return dest / "model.safetensors"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs", nargs="+", required=True, help="run names or local .safetensors"
    )
    parser.add_argument("--repo", default="minhleduc/rlcd-e2-checkpoints")
    parser.add_argument(
        "--cache", default=str(REPO_ROOT / "results" / "e2_vm" / "weights")
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--n_cases", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--tta", action="store_true", help="add option-order TTA")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    device = detect_device()
    hf = load_dataset("LocalLLaMA/typed-decisions", "all", split=args.split)
    if args.n_cases is not None:
        hf = hf.select(range(min(len(hf), args.n_cases)))

    normal: dict[str, list] = {}
    tuned: dict[str, list] = {}
    labels = None
    for run in args.runs:
        ckpt = (
            run
            if run.endswith(".safetensors")
            else resolve_weights(run, args.repo, args.cache)
        )
        model, cfg, tokenizer = load_model(Path(ckpt), device)
        ds = ViewDataset(hf, tokenizer, cfg.max_len, cfg.head_max_len, reverse=False)
        probs, labels, _ = predict_probs(model, ds, device, args.batch_size)
        normal[run] = probs
        if args.tta:
            rev_ds = ViewDataset(
                hf, tokenizer, cfg.max_len, cfg.head_max_len, reverse=True
            )
            rev_probs, _, flags = predict_probs(model, rev_ds, device, args.batch_size)
            tuned[run] = mean_probs([probs, unflip(rev_probs, flags)])
        else:
            tuned[run] = probs
        print(
            f"{run}: acc={accuracy(probs, labels):.4f}"
            + (f"  TTA={accuracy(tuned[run], labels):.4f}" if args.tta else "")
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result = {
        "runs": args.runs,
        "tta": args.tta,
        "single": {r: accuracy(p, labels) for r, p in normal.items()},
        "single_tta": {r: accuracy(p, labels) for r, p in tuned.items()}
        if args.tta
        else None,
        "ensemble": accuracy(mean_probs(list(normal.values())), labels),
        "ensemble_tta": accuracy(mean_probs(list(tuned.values())), labels)
        if args.tta
        else None,
    }
    print(json.dumps(result, indent=2))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
