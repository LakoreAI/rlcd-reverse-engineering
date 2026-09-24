import numpy as np
import pytest
import torch

from src.data import (
    FeatureDataset,
    dataset_from_npz,
    load_npz,
    make_synthetic_data,
    save_npz,
    split_train_val,
)


def test_make_synthetic_data_shapes():
    x, y = make_synthetic_data(n_samples=50, n_features=8, n_classes=3, seed=0)
    assert x.shape == (50, 8)
    assert y.shape == (50,)
    assert x.dtype == np.float32
    assert set(np.unique(y)) <= {0, 1, 2}


def test_save_load_roundtrip(tmp_path):
    x, y = make_synthetic_data(n_samples=20, n_features=5, n_classes=2, seed=1)
    path = tmp_path / "split.npz"
    save_npz(path, x, y)
    x2, y2 = load_npz(path)
    assert x2.shape == x.shape and y2.shape == y.shape
    assert x2.dtype == torch.float32 and y2.dtype == torch.int64
    assert torch.allclose(x2, torch.as_tensor(x))


def test_feature_dataset_properties():
    x = torch.randn(10, 7)
    y = torch.tensor([0, 1, 2] * 3 + [2])
    ds = FeatureDataset(x, y)
    assert len(ds) == 10
    assert ds.input_dim == 7
    assert ds.num_classes == 3
    xi, yi = ds[0]
    assert xi.shape == (7,) and yi.dim() == 0


def test_dataset_from_npz(tmp_path):
    x, y = make_synthetic_data(n_samples=30, n_features=6, n_classes=2, seed=2)
    path = tmp_path / "t.npz"
    save_npz(path, x, y)
    ds = dataset_from_npz(path)
    assert len(ds) == 30


def test_split_train_val_deterministic():
    x = torch.randn(100, 4)
    y = torch.randint(0, 2, (100,))
    ds = FeatureDataset(x, y)
    train1, val1 = split_train_val(ds, 0.2, seed=0)
    train2, val2 = split_train_val(ds, 0.2, seed=0)
    assert val1 is not None and len(val1) == 20
    assert len(train1) == 80
    assert sorted(val1.indices) == sorted(val2.indices)


def test_split_train_val_disabled():
    ds = FeatureDataset(torch.randn(10, 4), torch.zeros(10, dtype=torch.long))
    train, val = split_train_val(ds, 0.0, seed=0)
    assert val is None
    assert len(train) == 10


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_npz(tmp_path / "nope.npz")


def test_bad_arrays_raise(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(path, X=np.zeros((4, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        load_npz(path)
