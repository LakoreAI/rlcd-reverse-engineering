"""Model architecture for the reference MLP classifier.

This is the ARCHITECTURE-ONLY config — layer widths, activations, and
regularization. Training-loop hyperparameters (epochs, lr, data paths,
callback wiring) live in `src.pipelines.config.TrainingConfig`, mirroring the
split used by the reference research codebase this template is modelled on.

Replace this file's `MLPConfig` with your own model's config; keep the split so
checkpoints can rebuild the architecture without the training YAML.
"""

from dataclasses import dataclass
from typing import Tuple

# Activation name -> nn.Module class resolved in src.modules.model. Kept as
# strings here so this dataclass stays free of torch imports and is cheap to
# serialize into a checkpoint.
SUPPORTED_ACTIVATIONS = ("relu", "gelu", "tanh", "silu")


@dataclass
class MLPConfig:
    """A stack of fully-connected blocks plus a linear classification head.

    Layout:

        x (input_dim) -> [Linear -> (BatchNorm) -> Activation -> Dropout] * n
                      -> embed_dim -> Linear -> num_classes

    The penultimate `embed_dim` vector is returned alongside the logits so
    downstream code can use the representation (retrieval, clustering, a
    different head) without re-running the backbone.
    """

    # --- classifier head ---
    num_classes: int = 0  # set at runtime from the dataset
    embed_dim: int = 128  # width of the penultimate feature vector

    # --- input ---
    input_dim: int = 0  # set at runtime from the dataset

    # --- backbone ---
    hidden_dims: Tuple[int, ...] = (256, 256)
    dropout: float = 0.1
    activation: str = "relu"
    batch_norm: bool = True

    def __post_init__(self) -> None:
        if self.num_classes < 0:
            raise ValueError(f"num_classes must be >= 0, got {self.num_classes}")
        if self.input_dim < 0:
            raise ValueError(f"input_dim must be >= 0, got {self.input_dim}")
        if self.embed_dim <= 0:
            raise ValueError(f"embed_dim must be > 0, got {self.embed_dim}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.activation not in SUPPORTED_ACTIVATIONS:
            raise ValueError(
                f"unknown activation {self.activation!r}; "
                f"choose from {SUPPORTED_ACTIVATIONS}"
            )
