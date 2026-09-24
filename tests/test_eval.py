from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.config import DecisionModelConfig
from src.data import collate_fn
from src.modules.model import DecisionModel
from src.pipelines.eval import (
    apply_temperature,
    collect_rows,
    eval_per_epoch,
    evaluate,
    expected_calibration_error,
    fit_temperature,
    fit_temperatures,
    format_report,
    k_bucket,
    raw_metrics,
)


def test_k_bucket_boundaries():
    assert k_bucket(2) == 0
    assert k_bucket(3) == 1
    assert k_bucket(5) == 1
    assert k_bucket(6) == 2
    assert k_bucket(10) == 2
    assert k_bucket(11) == 3
    assert k_bucket(77) == 3


def test_ece_is_zero_for_perfectly_calibrated_confidences():
    # confidence exactly equals the empirical accuracy in each bin: 9/10
    # correct at confidence 0.9, 1/10 correct at confidence 0.1.
    confidences = torch.tensor([0.9] * 10 + [0.1] * 10)
    correct = torch.tensor([1] * 9 + [0] * 1 + [1] * 1 + [0] * 9)
    assert expected_calibration_error(confidences, correct) < 1e-6


def test_ece_is_positive_for_overconfident_predictions():
    confidences = torch.full((10,), 0.9)
    correct = torch.tensor([1] * 5 + [0] * 5)
    assert expected_calibration_error(confidences, correct) > 0.3


def make_row(qtype: int, k: int, seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(k, generator=g)
    target = torch.softmax(torch.randn(k, generator=g), dim=-1)
    return {"logits": logits, "target": target, "qtype": qtype, "k": k}


def test_raw_metrics_shape_and_bounds():
    rows = [make_row(0, 4, seed=i) for i in range(20)]
    result = raw_metrics(rows)
    assert result["n"] == 20
    assert 0.0 <= result["accuracy"] <= 1.0
    assert 0.0 <= result["ece"] <= 1.0
    assert np.isfinite(result["brier"])
    assert np.isfinite(result["nll"])


def test_fit_temperature_recovers_scaling():
    # true_logits / T_true softmaxed matches target exactly; fitting from
    # the un-scaled true_logits should recover something close to T_true.
    torch.manual_seed(0)
    true_logits = torch.randn(200, 3) * 3.0
    t_true = 2.5
    target = torch.softmax(true_logits / t_true, dim=-1)
    mask = torch.ones(200, 3, dtype=torch.bool)

    fitted = fit_temperature(true_logits, target, mask)
    assert abs(fitted - t_true) < 0.3


def test_fit_temperatures_groups_by_qtype_and_bucket():
    rows = [make_row(0, 2, seed=i) for i in range(10)] + [
        make_row(1, 8, seed=i + 100) for i in range(10)
    ]
    temps = fit_temperatures(rows)
    assert set(temps) == {(0, k_bucket(2)), (1, k_bucket(8))}
    for t in temps.values():
        assert 0.5 <= t <= 5.0


def test_apply_temperature_changes_confidence():
    rows = [make_row(0, 4, seed=i) for i in range(20)]
    temps = {(0, k_bucket(4)): 3.0}
    raw = raw_metrics(rows)
    scaled = apply_temperature(rows, temps)
    # dividing sharp logits by T=3 should flatten (lower) average confidence
    assert scaled["ece"] != raw["ece"] or scaled["brier"] != raw["brier"]


class RowDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def make_encodable_row(qtype: int, k: int, seq_len: int = 12) -> dict:
    assert k <= seq_len
    ids = list(range(4, 4 + seq_len))
    markers = list(range(k))
    target = [1.0 / k] * k
    return {"ids": ids, "markers": markers, "target": target, "qtype": qtype}


def make_loader(
    n_rows: int, qtype: int, k: int, batch_size: int = 4, pad_token_id: int = 0
):
    rows = [make_encodable_row(qtype, k) for _ in range(n_rows)]
    return DataLoader(
        RowDataset(rows),
        batch_size=batch_size,
        collate_fn=partial(collate_fn, pad_token_id=pad_token_id),
    )


def test_collect_rows_and_eval_per_epoch(dummy_encoder):
    cfg = DecisionModelConfig(head_layers=1)
    model = DecisionModel(cfg, encoder=dummy_encoder).eval()
    loader = make_loader(n_rows=8, qtype=0, k=3)

    rows = collect_rows(model, loader, torch.device("cpu"))
    assert len(rows) == 8
    for row in rows:
        assert row["k"] == 3
        assert row["logits"].shape == (3,)

    extra = eval_per_epoch(model, loader, torch.device("cpu"))
    assert set(extra) == {"raw_ece", "raw_brier", "raw_nll", "raw_accuracy", "raw_n"}


def test_eval_per_epoch_empty_loader_returns_empty_dict(dummy_encoder):
    cfg = DecisionModelConfig(head_layers=1)
    model = DecisionModel(cfg, encoder=dummy_encoder).eval()
    assert eval_per_epoch(model, None, torch.device("cpu")) == {}


def test_evaluate_end_to_end_produces_report(dummy_encoder):
    cfg = DecisionModelConfig(head_layers=1)
    model = DecisionModel(cfg, encoder=dummy_encoder).eval()
    calib_loader = make_loader(n_rows=16, qtype=0, k=4)
    test_loader = make_loader(n_rows=12, qtype=0, k=4)

    result = evaluate(model, calib_loader, test_loader, torch.device("cpu"))
    assert set(result) == {"raw", "post_temperature", "fitted_temperature"}
    assert result["raw"]["n"] == 12
    assert result["post_temperature"]["n"] == 12

    report = format_report(result)
    assert "raw:" in report
    assert "postT:" in report
