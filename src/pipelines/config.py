"""Training-loop hyperparameters — separate from `src.config.DecisionModelConfig`
(model architecture only). Controllable via a YAML file (see
`configs/train.yaml` / `configs/rlcd_smoke.yaml`) with individual fields
overridable from the CLI.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Union

from src.utils.io_utils import read_yaml


@dataclass
class TrainingConfig:
    # --- data ---
    # `LocalLLaMA/typed-decisions` ships train/test HF splits directly; the
    # calibration slice is carved out of `train` at load time (never from
    # `test` — fitting temperature on training items inflates it, per
    # Laya's own issue #186).
    dataset_name: str = "LocalLLaMA/typed-decisions"
    dataset_config: str = "all"
    calib_fraction: float = 0.1
    seed: int = 42
    # Caps train/test to this many cases (applied before the calib split) —
    # a quick pipeline sanity check on a small slice, not a real run.
    max_examples: Optional[int] = None
    # Randomly permute the option order of choice/noul questions on each
    # training draw (label-preserving: the target/label follow the permuted
    # keys). Score levels are ordinal and left alone. A cheap order-invariance
    # augmenter / regularizer for the small training set; never applied to the
    # calibration or test datasets.
    augment_permute: bool = False

    # --- output ---
    ckpt_dir: str = "checkpoints"
    result_dir: str = "results"
    run_name: Optional[str] = None

    # --- optimization ---
    epochs: int = 10
    batch_size: int = 16
    accum_steps: int = 1
    num_workers: int = 0
    pin_memory: bool = False
    # Mixed precision. "bf16" (Ampere+ CUDA, or Apple MPS) needs no loss
    # scaling; "fp16" (e.g. T4) uses a GradScaler, CUDA only. Ignored on CPU.
    amp: bool = False
    amp_dtype: str = "bf16"
    gradient_checkpointing: bool = False
    lr: float = 2e-5
    # Separate LR for the non-encoder params (head/type_emb/scorer); None
    # uses `lr` for everything. Laya's notebook: 2.5e-5 encoder, 1e-4 head.
    lr_head: Optional[float] = None
    weight_decay: float = 0.01
    grad_clip: Optional[float] = None
    log_every: int = 10
    eval_every: int = 1
    ckpt_every: int = 1

    # --- RLCD loss (Laya's training step: RL term + soft-target CE term) ---
    # loss = w_rl * L_rl + w_ce * L_ce. CE-only: w_rl=0. RL-only: w_ce=0.
    w_rl: float = 1.0
    w_ce: float = 1.0
    num_noise_samples: int = 4  # number of noisy logit samples per step
    sigma_start: float = 1.0
    sigma_end: float = 0.1
    # True anneals sigma_start -> sigma_end over training (Laya default);
    # false holds sigma fixed at sigma_start (loss-ablation fixed-sigma runs).
    anneal_sigma: bool = True
    reward_w_spherical: float = 0.5
    reward_w_rps: float = 1.0

    # Optional DecisionModelConfig architecture overrides, e.g.
    # {"encoder_name": "google/bert_uncased_L-2_H-128_A-2", "head_layers": 1}.
    arch: Optional[dict] = None

    # Initialise the full DecisionModel (encoder + head + scorer) from a Laya
    # checkpoint instead of the bare pretrained encoder: an HF repo id
    # (e.g. "convaiinnovations/laya", reads its `model.safetensors`) or a
    # local .safetensors path. Laya's `act_head` and scalar `temperature`
    # are dropped (see src/modules/model.py).
    init_from: Optional[str] = None

    # E3: every `log_every` steps, log cosine(∇RL, ∇CE) and ||∇RL||/||∇CE||
    # taken w.r.t. the logits (not the parameters — logit-level is free,
    # parameter-level would cost two extra backward passes).
    log_grad_diagnostics: bool = False

    # --- callbacks ---
    # None (or {"type": "none"}) disables LR scheduling. See
    # src/callbacks/lr_scheduler.py.
    lr_scheduler: Optional[dict] = None
    early_stopping: Optional[dict] = None
    save_best: bool = True
    # Which validation metric BestCheckpoint/early stopping monitor. Raw
    # (pre-temperature) ECE is minimized — see src/pipelines/eval.py.
    best_metric: str = "raw_ece"
    best_mode: str = "min"

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
