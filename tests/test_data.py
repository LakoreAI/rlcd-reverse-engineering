import json

import pytest
from datasets import Dataset as HFDataset

from src.config import QTYPES
from src.data import (
    TypedDecisionDataset,
    build_sequence,
    collate_fn,
    label_index,
    option_keys,
    render_options,
    split_train_calib,
    target_vector,
)

CHOICE_Q = {
    "type": "choice",
    "instructions": "What should happen?",
    "criteria": {"stop": "Halt now.", "go": "Proceed."},
}
SCORE_Q = {
    "type": "score",
    "instructions": "How risky is this?",
    "criteria": ["Benign.", "Low.", "Moderate.", "High."],
}
NOUL_Q = {
    "type": "noul",
    "instructions": "Does this need review?",
    "criteria": {"false": "No.", "true": "Yes."},
}


def make_hf_case(case_id: str, questions: dict, gold: dict) -> dict:
    return {
        "id": case_id,
        "state": json.dumps({"task": "do the thing"}),
        "questions": json.dumps(questions),
        "gold": json.dumps(gold),
    }


def test_render_options_choice_preserves_key_order():
    assert render_options(CHOICE_Q) == ["stop: Halt now.", "go: Proceed."]


def test_render_options_score_uses_ordinal_levels():
    opts = render_options(SCORE_Q)
    assert opts[0] == "level 0: Benign."
    assert opts[-1] == "level 3: High."


def test_render_options_noul_uses_criteria_descriptions():
    assert render_options(NOUL_Q) == ["false: No.", "true: Yes."]


def test_render_options_noul_without_criteria_falls_back_to_default():
    # LocalLLaMA/typed-decisions sometimes omits `criteria` for noul
    # questions (a plain yes/no with no dataset-provided wording).
    bare_noul = {"type": "noul", "instructions": "Does X hold?"}
    opts = render_options(bare_noul)
    assert len(opts) == 2
    assert option_keys(bare_noul) == ["false", "true"]


def test_option_keys_matches_render_options_length():
    for q in (CHOICE_Q, SCORE_Q, NOUL_Q):
        assert len(option_keys(q)) == len(render_options(q))


def test_render_options_rejects_unknown_type():
    with pytest.raises(ValueError):
        render_options({"type": "essay", "instructions": "", "criteria": {}})


def test_target_vector_aligned_to_option_keys():
    gold = {"probabilities": {"stop": 0.3, "go": 0.7}}
    assert target_vector(CHOICE_Q, gold) == [0.3, 0.7]


def test_target_vector_score_indexes_by_level():
    gold = {"probabilities": {"0": 0.1, "1": 0.2, "2": 0.3, "3": 0.4}}
    assert target_vector(SCORE_Q, gold) == [0.1, 0.2, 0.3, 0.4]


def test_build_sequence_markers_point_at_mask_tokens(dummy_tokenizer):
    ids, markers = build_sequence(dummy_tokenizer, "some state text", CHOICE_Q)
    assert len(markers) == len(render_options(CHOICE_Q))
    for m in markers:
        assert ids[m] == dummy_tokenizer.mask_token_id


def test_build_sequence_starts_and_bounds(dummy_tokenizer):
    ids, _markers = build_sequence(dummy_tokenizer, "state", CHOICE_Q, max_len=64)
    assert ids[0] == dummy_tokenizer.cls_token_id
    assert ids[-1] == dummy_tokenizer.sep_token_id
    assert len(ids) <= 64


def test_build_sequence_truncates_under_tight_option_budget(dummy_tokenizer):
    many_options_q = {
        "type": "choice",
        "instructions": "Pick one of many.",
        "criteria": {
            f"opt{i}": f"a fairly long description of option {i}" for i in range(20)
        },
    }
    ids, markers = build_sequence(
        dummy_tokenizer, "state", many_options_q, head_max_len=32
    )
    assert len(markers) == 20
    # every marker must still land inside the sequence and on a mask token
    for m in markers:
        assert m < len(ids)
        assert ids[m] == dummy_tokenizer.mask_token_id


def test_typed_decision_dataset_flattens_one_row_per_question(dummy_tokenizer):
    case = make_hf_case(
        "c0",
        {"q1": CHOICE_Q, "q2": SCORE_Q},
        {
            "q1": {"probabilities": {"stop": 0.4, "go": 0.6}},
            "q2": {"probabilities": {"0": 0.25, "1": 0.25, "2": 0.25, "3": 0.25}},
        },
    )
    hf_ds = HFDataset.from_list([case])
    ds = TypedDecisionDataset(hf_ds, dummy_tokenizer, max_len=64, head_max_len=32)
    assert len(ds) == 2

    row0 = ds[0]
    assert set(row0) == {"ids", "markers", "target", "qtype", "label"}
    assert row0["qtype"] == QTYPES["choice"]
    assert len(row0["target"]) == 2

    row1 = ds[1]
    assert row1["qtype"] == QTYPES["score"]
    assert len(row1["target"]) == 4


def test_collate_fn_pads_variable_option_counts(dummy_tokenizer):
    case = make_hf_case(
        "c0",
        {"q1": CHOICE_Q, "q2": SCORE_Q},
        {
            "q1": {"probabilities": {"stop": 0.4, "go": 0.6}},
            "q2": {"probabilities": {"0": 0.25, "1": 0.25, "2": 0.25, "3": 0.25}},
        },
    )
    hf_ds = HFDataset.from_list([case])
    ds = TypedDecisionDataset(hf_ds, dummy_tokenizer, max_len=64, head_max_len=32)
    batch = collate_fn([ds[0], ds[1]], pad_token_id=dummy_tokenizer.pad_token_id)

    assert batch["target"].shape[0] == 2
    assert batch["marker_mask"].shape == batch["marker_pos"].shape
    # row 0 (choice, 2 options) padded to row 1's 4 options
    assert batch["marker_mask"][0].sum() == 2
    assert batch["marker_mask"][1].sum() == 4
    assert not batch["marker_mask"][0, 2:].any()


def test_split_train_calib_deterministic():
    cases = [
        make_hf_case(
            f"c{i}",
            {"q1": CHOICE_Q},
            {"q1": {"probabilities": {"stop": 0.5, "go": 0.5}}},
        )
        for i in range(50)
    ]
    hf_ds = HFDataset.from_list(cases)

    train1, calib1 = split_train_calib(hf_ds, calib_fraction=0.2, seed=0)
    train2, calib2 = split_train_calib(hf_ds, calib_fraction=0.2, seed=0)

    assert len(calib1) == 10
    assert len(train1) == 40
    assert list(calib1["id"]) == list(calib2["id"])


def test_split_train_calib_disabled_returns_none():
    cases = [
        make_hf_case(
            "c0", {"q1": CHOICE_Q}, {"q1": {"probabilities": {"stop": 0.5, "go": 0.5}}}
        )
    ]
    hf_ds = HFDataset.from_list(cases)
    train, calib = split_train_calib(hf_ds, calib_fraction=0.0, seed=0)
    assert calib is None
    assert len(train) == 1


def test_label_index_matches_option_keys():
    assert label_index(CHOICE_Q, {"label": "go"}) == 1
    assert label_index(SCORE_Q, {"label": "2"}) == 2
    assert label_index(NOUL_Q, {"label": "True"}) == 1
    assert label_index(CHOICE_Q, {}) == -1
    assert label_index(CHOICE_Q, {"label": "unknown"}) == -1


def test_collate_fn_carries_labels(dummy_tokenizer):
    case = make_hf_case(
        "c0",
        {"q1": CHOICE_Q},
        {"q1": {"label": "go", "probabilities": {"stop": 0.4, "go": 0.6}}},
    )
    ds = TypedDecisionDataset(
        HFDataset.from_list([case]), dummy_tokenizer, max_len=64, head_max_len=32
    )
    batch = collate_fn([ds[0]], pad_token_id=dummy_tokenizer.pad_token_id)
    assert batch["label"].tolist() == [1]
