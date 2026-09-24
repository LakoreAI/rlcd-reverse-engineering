"""Reference model: an MLP backbone with a linear classification head.

    x (B, input_dim)
      -> MLPBackbone            -> feature (B, embed_dim)
      -> nn.Linear(embed_dim, num_classes) -> logits (B, num_classes)

`forward` returns `(logits, feature)` and accepts an optional `label` argument
that is currently unused. It is kept in the signature so a margin/ArcFace-style
head can replace the plain linear head later without touching the training
loop, the evaluator, or the tests.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.config import MLPConfig


def _make_activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    if name == "silu":
        return nn.SiLU(inplace=True)
    raise ValueError(f"unknown activation: {name!r}")


class MLPBackbone(nn.Module):
    """`input_dim -> hidden_dims -> embed_dim` with BN, activation, dropout."""

    def __init__(self, cfg: MLPConfig):
        super().__init__()
        layers = []
        in_dim = cfg.input_dim
        for hidden_dim in cfg.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if cfg.batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(_make_activation(cfg.activation))
            if cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden_dim

        # Project the last hidden width to the requested embedding width. When
        # hidden_dims is empty this is just a Linear(input_dim, embed_dim).
        if in_dim != cfg.embed_dim or not cfg.hidden_dims:
            layers.append(nn.Linear(in_dim, cfg.embed_dim))
            if cfg.batch_norm:
                layers.append(nn.BatchNorm1d(cfg.embed_dim))
            layers.append(_make_activation(cfg.activation))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPClassifier(nn.Module):
    def __init__(self, cfg: MLPConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.num_classes <= 0:
            raise ValueError(
                "MLPConfig.num_classes must be set (> 0) before building the model"
            )
        self.backbone = MLPBackbone(cfg)
        self.head = nn.Linear(cfg.embed_dim, cfg.num_classes)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Feature vector alone — useful for retrieval / calibration paths."""
        return self.backbone(x)

    def forward(
        self,
        x: torch.Tensor,
        label: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: (B, input_dim). `label` is accepted for interface parity with
        margin-based heads and is currently unused.

        Returns (logits (B, num_classes), feature (B, embed_dim)).
        """
        feature = self.backbone(x)
        logits = self.head(feature)
        return logits, feature
