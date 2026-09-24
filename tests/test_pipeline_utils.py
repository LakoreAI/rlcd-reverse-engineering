from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from src.config import MLPConfig
from src.data import FeatureDataset, make_synthetic_data
from src.modules.model import MLPClassifier
from src.pipelines._utils import announce_training, label_counts
from src.pipelines.config import TrainingConfig


def make_dataset(n=24, d=6, c=3) -> FeatureDataset:
    x, y = make_synthetic_data(n_samples=n, n_features=d, n_classes=c, seed=0)
    return FeatureDataset(torch.as_tensor(x), torch.as_tensor(y))


def test_label_counts_full_and_subset():
    ds = make_dataset()
    assert int(label_counts(ds).sum()) == len(ds)
    sub = Subset(ds, list(range(10)))
    assert int(label_counts(sub).sum()) == 10


def test_label_counts_unlabeled_returns_none():
    class Plain(Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, i):
            return torch.zeros(3), 0

    assert label_counts(Plain()) is None


def test_announce_training_prints_config_data_and_model(capsys):
    ds = make_dataset()
    loader = DataLoader(ds, batch_size=8)
    cfg = MLPConfig(input_dim=6, num_classes=3)
    model = MLPClassifier(cfg)

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
    assert "model config (MLPConfig)" in out
    assert "training config (TrainingConfig)" in out
    assert "data: train sample" in out
    assert "TOTAL" in out
    assert "forward check (untrained)" in out
    assert "logits" in out
