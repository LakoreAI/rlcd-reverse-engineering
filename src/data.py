"""Dataset plumbing for fixed-size feature vectors.

One example is `(x, label)`: an `(input_dim,)` float32 feature vector and an
integer class label. Features are stored on disk as `.npz` archives with two
arrays:

    X : (N, input_dim) float32
    y : (N,)           int64

Replace `FeatureDataset` / `load_npz` when bringing your own input format; the
rest of the pipeline only needs a `Dataset` yielding `(x, label)` plus the
`input_dim` / `num_classes` properties.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Subset


def load_npz(path: str | Path) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load an `.npz` with `X` / `y` into `(float32 X, int64 y)` tensors."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"feature file not found: {path}")
    with np.load(path) as data:
        if "X" not in data or "y" not in data:
            raise ValueError(f"{path} must contain 'X' and 'y' arrays")
        x = torch.as_tensor(np.asarray(data["X"], dtype=np.float32))
        y = torch.as_tensor(np.asarray(data["y"], dtype=np.int64))
    if x.dim() != 2:
        raise ValueError(f"X must be 2-D (N, input_dim), got shape {tuple(x.shape)}")
    if y.dim() != 1 or y.shape[0] != x.shape[0]:
        raise ValueError(
            f"y must be 1-D with one label per row; got X{tuple(x.shape)} y{tuple(y.shape)}"
        )
    return x, y


def save_npz(path: str | Path, x, y) -> None:
    """Write feature/label arrays to an `.npz` archive."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, X=np.asarray(x, dtype=np.float32), y=np.asarray(y, dtype=np.int64))


class FeatureDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same number of rows")
        self.x = x.float()
        self.y = y.long()

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, index: int):
        return self.x[index], self.y[index]

    @property
    def input_dim(self) -> int:
        return int(self.x.shape[1])

    @property
    def num_classes(self) -> int:
        return int(self.y.max().item()) + 1 if len(self) else 0


def dataset_from_npz(path: str | Path) -> FeatureDataset:
    x, y = load_npz(path)
    return FeatureDataset(x, y)


def make_synthetic_data(
    n_samples: int = 2000,
    n_features: int = 32,
    n_classes: int = 5,
    n_informative: int = 10,
    class_sep: float = 1.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Linearly separable-ish synthetic classification data.

    Used by the smoke test and by anyone who wants to exercise the pipeline
    before wiring in a real dataset. `sklearn` is imported lazily so the core
    training path does not require it.
    """
    from sklearn.datasets import make_classification

    x, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=min(n_informative, n_features),
        n_redundant=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        class_sep=class_sep,
        flip_y=0.01,
        random_state=seed,
    )
    return x.astype(np.float32), y.astype(np.int64)


def split_train_val(
    dataset: FeatureDataset, val_fraction: float, seed: int
) -> Tuple[Dataset, Dataset | None]:
    """Deterministic random split into `(train, val)` subsets.

    Returns `(dataset, None)` when `val_fraction <= 0` or the dataset is too
    small to split. Splits are index-based `Subset`s, so features stay shared.
    """
    n = len(dataset)
    if val_fraction <= 0 or n < 2:
        return dataset, None
    n_val = max(int(round(n * val_fraction)), 1)
    n_val = min(n_val, n - 1)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return Subset(dataset, train_idx), Subset(dataset, val_idx)
