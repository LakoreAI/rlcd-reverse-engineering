"""Training-loop hyperparameters — separate from `src.config.MLPConfig`
(model architecture only). Controllable via a YAML file (see
`configs/train.yaml`) with individual fields overridable from the CLI.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Union

from src.utils.io_utils import read_yaml


@dataclass
class TrainingConfig:
    # --- data ---
    # `data_root/<train_file>` is required; `val_file` / `test_file` are
    # optional and skipped when absent.
    data_root: str = "data/raw"
    train_file: str = "train.npz"
    val_file: Optional[str] = "val.npz"
    test_file: Optional[str] = "test.npz"
    # If no val_file is present, hold out this fraction of the train set. 0
    # disables the held-out split (best-checkpointing then uses test metrics).
    val_fraction: float = 0.0
    seed: int = 42

    # --- output ---
    ckpt_dir: str = "checkpoints"
    result_dir: str = "results"
    run_name: Optional[str] = None

    # --- optimization ---
    epochs: int = 100
    batch_size: int = 128
    accum_steps: int = 1
    num_workers: int = 0
    pin_memory: bool = False
    amp: bool = False
    lr: float = 1e-3
    weight_decay: float = 0.0
    log_every: int = 10
    eval_every: int = 1
    ckpt_every: int = 10

    # Optional MLPConfig architecture overrides, e.g.
    # {"hidden_dims": [512, 256], "dropout": 0.2, "embed_dim": 256}.
    # Keys must be MLPConfig fields (input_dim/num_classes are set at runtime).
    arch: Optional[dict] = None

    # --- callbacks ---
    # None (or {"type": "none"}) disables LR scheduling. See
    # src/callbacks/lr_scheduler.py.
    lr_scheduler: Optional[dict] = None
    early_stopping: Optional[dict] = None
    save_best: bool = True
    # Which validation metric BestCheckpoint/early stopping monitor.
    # "accuracy"/"f1" are maximized; "val_loss" is minimized.
    best_metric: str = "accuracy"
    best_mode: str = "max"

    # None (or {"enabled": false}) disables W&B logging.
    wandb: Optional[dict] = None

    resume_from: Optional[str] = None


def load_training_config(
    config_path: Optional[Union[str, Path]] = None, **cli_overrides
) -> TrainingConfig:
    """Build a TrainingConfig from defaults, a YAML file, then CLI overrides
    (highest precedence, but only applied for keys the caller actually passed —
    argparse defaults of None are treated as "not set").
    """
    cfg = TrainingConfig()

    if config_path is not None:
        yaml_data = read_yaml(config_path)
        known_fields = set(asdict(cfg).keys())
        for key, value in yaml_data.items():
            if key not in known_fields:
                raise ValueError(
                    f"unknown training config key {key!r} in {config_path}"
                )
            setattr(cfg, key, value)

    for key, value in cli_overrides.items():
        if value is not None:
            setattr(cfg, key, value)

    return cfg
