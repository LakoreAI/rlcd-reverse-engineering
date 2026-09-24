"""Training loss and the logit-derived helpers used at evaluation time."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationLoss(nn.Module):
    """Plain cross-entropy over class logits."""

    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.ce(logits, labels)


@torch.no_grad()
def probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Softmax probabilities, (B, num_classes)."""
    return F.softmax(logits, dim=1)


@torch.no_grad()
def predict(logits: torch.Tensor) -> torch.Tensor:
    """Argmax class predictions, (B,)."""
    return logits.argmax(dim=1)
