import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import MLPConfig
from src.data import FeatureDataset, make_synthetic_data
from src.modules.model import MLPClassifier
from src.pipelines.eval import eval_per_epoch, evaluate, format_report


def make_loader(n=200, input_dim=12, num_classes=3, seed=0):
    x, y = make_synthetic_data(
        n_samples=n, n_features=input_dim, n_classes=num_classes, seed=seed
    )
    ds = FeatureDataset(torch.as_tensor(x), torch.as_tensor(y))
    return DataLoader(ds, batch_size=32, shuffle=False), num_classes, input_dim


def test_evaluate_returns_finite_metrics():
    loader, num_classes, input_dim = make_loader()
    cfg = MLPConfig(input_dim=input_dim, num_classes=num_classes)
    torch.manual_seed(0)
    model = MLPClassifier(cfg).eval()

    result = evaluate(model, loader, torch.device("cpu"), num_classes=num_classes)
    assert np.isfinite(result["accuracy"])
    assert np.isfinite(result["f1"])
    assert np.isfinite(result["loss"])
    assert 0.0 <= result["accuracy"] <= 1.0
    assert 0.0 <= result["f1"] <= 1.0
    assert len(result["confusion"]) == num_classes
    assert set(result["per_class"]) == set(range(num_classes))


def test_evaluate_delegates_to_eval_per_epoch():
    loader, num_classes, input_dim = make_loader(n=120, seed=3)
    cfg = MLPConfig(input_dim=input_dim, num_classes=num_classes)
    torch.manual_seed(0)
    model = MLPClassifier(cfg).eval()

    per_epoch = eval_per_epoch(
        model, loader, torch.device("cpu"), num_classes=num_classes
    )
    wrapped = evaluate(model, loader, torch.device("cpu"), num_classes=num_classes)
    assert per_epoch == wrapped


def test_confusion_rows_sum_to_support():
    loader, num_classes, input_dim = make_loader(n=90)
    cfg = MLPConfig(input_dim=input_dim, num_classes=num_classes)
    model = MLPClassifier(cfg).eval()
    result = evaluate(model, loader, torch.device("cpu"), num_classes=num_classes)
    counts = np.asarray(result["confusion"]).sum(axis=1)
    assert counts.sum() == 90


def test_format_report_mentions_metrics():
    loader, num_classes, input_dim = make_loader(n=64)
    cfg = MLPConfig(input_dim=input_dim, num_classes=num_classes)
    model = MLPClassifier(cfg).eval()
    result = evaluate(model, loader, torch.device("cpu"), num_classes=num_classes)
    report = format_report(result)
    assert "accuracy" in report
    assert "macro-F1" in report
