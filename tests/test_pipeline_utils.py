import json
from functools import partial
from pathlib import Path

import torch
from datasets import Dataset as HFDataset
from torch.utils.data import DataLoader

from src.config import DecisionModelConfig
from src.data import TypedDecisionDataset, collate_fn
from src.modules.model import DecisionModel
from src.pipelines._utils import announce_training, qtype_counts
from src.pipelines.config import TrainingConfig

CHOICE_Q = {
    "type": "choice",
    "instructions": "What should happen?",
    "criteria": {"stop": "Halt now.", "go": "Proceed."},
}
SCORE_Q = {
    "type": "score",
    "instructions": "How risky is this?",
    "criteria": ["Benign.", "Low."],
}


def make_dataset(dummy_tokenizer) -> TypedDecisionDataset:
    cases = [
        {
            "id": "c0",
            "state": json.dumps({"task": "do the thing"}),
            "questions": json.dumps({"q1": CHOICE_Q, "q2": SCORE_Q}),
            "gold": json.dumps(
                {
                    "q1": {"probabilities": {"stop": 0.4, "go": 0.6}},
                    "q2": {"probabilities": {"0": 0.5, "1": 0.5}},
                }
            ),
        }
    ]
    hf_ds = HFDataset.from_list(cases)
    return TypedDecisionDataset(hf_ds, dummy_tokenizer, max_len=64, head_max_len=32)


def test_qtype_counts(dummy_tokenizer):
    ds = make_dataset(dummy_tokenizer)
    counts = qtype_counts(ds)
    assert counts == {"choice": 1, "score": 1}


def test_announce_training_prints_config_data_and_model(capsys, dummy_tokenizer, dummy_encoder):
    ds = make_dataset(dummy_tokenizer)
    loader = DataLoader(
        ds, batch_size=2, collate_fn=partial(collate_fn, pad_token_id=dummy_tokenizer.pad_token_id)
    )
    cfg = DecisionModelConfig(head_layers=1)
    model = DecisionModel(cfg, encoder=dummy_encoder)

    announce_training(
        model=model,
        model_cfg=cfg,
        train_cfg=TrainingConfig(seed=7),
        train_loader=loader,
        train_dataset=ds,
        device=torch.device("cpu"),
        run_name="run-x",
        ckpt_dir=Path("ckpt"),
        result_dir=Path("res"),
    )
    out = capsys.readouterr().out
    assert "RUN  run-x" in out
    assert "model config (DecisionModelConfig)" in out
    assert "training config (TrainingConfig)" in out
    assert "data: train sample" in out
    assert "TOTAL" in out
    assert "forward check (untrained)" in out
    assert "logits" in out
