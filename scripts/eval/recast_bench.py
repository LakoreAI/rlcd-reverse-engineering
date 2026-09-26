"""Evaluate a typed-decision model zero-shot on recast text-classification tasks.

A classification dataset is recast into one `choice` question per example:
the state is the text, the options are the dataset's class names, and the
prediction is the argmax option. This is the same recasting Laya's own
cross-task numbers use (AG News, Emotion, Banking77, ...), so it lets the open
models be compared against Laya's published figures and — crucially for this
report — lets the option-token budget be varied on a high-cardinality task
(Banking77, 77 intents).

Loads either:
  - `convaiinnovations/laya` / `...-typed-decisions` (encoder/head/scorer;
    their unused act_head and scalar temperature are dropped), or
  - a local/HF `.safetensors` with a `model_config.json` (our checkpoints).

    uv run python scripts/eval/recast_bench.py \
        --model convaiinnovations/laya --task banking77 --limit 300
    uv run python scripts/eval/recast_bench.py \
        --model /path/to/model.safetensors --task ag_news --json out.json
"""

import argparse
import json
import sys
from functools import partial
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.config import QTYPES, DecisionModelConfig  # noqa: E402
from src.data import build_sequence  # noqa: E402
from src.modules.model import DecisionModel  # noqa: E402
from src.pipelines.infer import load_model  # noqa: E402
from src.utils.model_utils import detect_device  # noqa: E402

TASKS = {
    "banking77": {
        "path": "legacy-datasets/banking77",
        "split": "test",
        "text": "text",
        "instruction": "Which banking intent does this customer message express?",
    },
    "ag_news": {
        "path": "fancyzhx/ag_news",
        "split": "test",
        "text": "text",
        "instruction": "Which news category does this article belong to?",
    },
    "emotion": {
        "path": "dair-ai/emotion",
        "split": "test",
        "text": "text",
        "instruction": "Which emotion does this text express?",
    },
    "sst5": {
        "path": "SetFit/sst5",
        "split": "test",
        "text": "text",
        "instruction": "What is the sentiment of this text?",
    },
}


class RecastDataset(Dataset):
    """One row per example: ids, marker positions, gold option index, K."""

    def __init__(self, records, names, instruction, tokenizer, max_len, head_max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.head_max_len = head_max_len
        self.rows = []
        for text, label in records:
            question = {
                "type": "choice",
                "instructions": instruction,
                "criteria": {str(i): name for i, name in enumerate(names)},
            }
            ids, markers = build_sequence(
                tokenizer, text, question, max_len, head_max_len
            )
            self.rows.append((ids, markers, int(label)))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        ids, markers, label = self.rows[index]
        return {"ids": ids, "markers": markers, "label": label}


def collate(batch, pad_token_id):
    b = len(batch)
    max_len = max(len(r["ids"]) for r in batch)
    max_opts = max(len(r["markers"]) for r in batch)
    ids = torch.full((b, max_len), pad_token_id, dtype=torch.long)
    attn = torch.zeros((b, max_len), dtype=torch.long)
    marker_pos = torch.zeros((b, max_opts), dtype=torch.long)
    marker_mask = torch.zeros((b, max_opts), dtype=torch.bool)
    label = torch.zeros(b, dtype=torch.long)
    for i, r in enumerate(batch):
        n = len(r["ids"])
        k = len(r["markers"])
        ids[i, :n] = torch.as_tensor(r["ids"])
        attn[i, :n] = 1
        marker_pos[i, :k] = torch.as_tensor(r["markers"])
        marker_mask[i, :k] = True
        label[i] = r["label"]
    return {
        "ids": ids,
        "attention_mask": attn,
        "marker_pos": marker_pos,
        "marker_mask": marker_mask,
        "qtype": torch.full((b,), QTYPES["choice"], dtype=torch.long),
        "label": label,
    }


def build_typed_model(cfg, device):
    model = DecisionModel(cfg).to(device)
    return model


def load_any_model(source: str, device, max_len, head_max_len):
    p = Path(source)
    if p.exists() and (p.suffix == ".safetensors" or p.name.endswith(".pt")):
        return load_model(p, device)
    # Laya checkpoints: no model_config.json; build the known Laya architecture
    # and load its encoder/head/scorer (act_head + scalar temperature dropped).
    from src.pipelines.train import load_laya_weights

    cfg = DecisionModelConfig(
        encoder_name="answerdotai/ModernBERT-large",
        head_layers=2,
        max_len=max_len,
        head_max_len=head_max_len,
    )
    model = build_typed_model(cfg, device)
    load_laya_weights(model, source)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.encoder_name)
    return model, cfg, tokenizer


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model", required=True, help="HF repo id or local .safetensors/.pt"
    )
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--split", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--head_max_len", type=int, default=256)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    spec = TASKS[args.task]
    device = detect_device()
    model, cfg, tokenizer = load_any_model(
        args.model, device, args.max_len, args.head_max_len
    )
    model.eval()

    ds = load_dataset(spec["path"], split=args.split or spec["split"])
    names = ds.features["label"].names
    if args.limit:
        ds = ds.select(range(min(len(ds), args.limit)))
    records = [(r[spec["text"]], r["label"]) for r in ds]
    dataset = RecastDataset(
        records, names, spec["instruction"], tokenizer, cfg.max_len, cfg.head_max_len
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=partial(collate, pad_token_id=tokenizer.pad_token_id),
    )

    correct = top5 = n = 0
    for batch in loader:
        logits = model(
            batch["ids"].to(device),
            batch["attention_mask"].to(device),
            batch["marker_pos"].to(device),
            batch["marker_mask"].to(device),
            batch["qtype"].to(device),
        ).cpu()
        label = batch["label"]
        for i in range(logits.shape[0]):
            k = int(batch["marker_mask"][i].sum())
            row = logits[i, :k]
            pred = int(row.argmax())
            correct += int(pred == int(label[i]))
            top5 += int(int(label[i]) in row.topk(min(5, k)).indices.tolist())
            n += 1

    result = {
        "model": args.model,
        "task": args.task,
        "n_classes": len(names),
        "n": n,
        "accuracy": correct / n,
        "top5": top5 / n,
    }
    print(json.dumps(result, indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
