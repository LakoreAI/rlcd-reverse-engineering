import pytest
import torch

from src.config import MLPConfig
from src.modules.model import MLPBackbone, MLPClassifier

B, INPUT_DIM, NUM_CLASSES = 8, 16, 4


def make_cfg(**overrides) -> MLPConfig:
    base = dict(input_dim=INPUT_DIM, num_classes=NUM_CLASSES)
    base.update(overrides)
    return MLPConfig(**base)


def test_forward_shapes():
    cfg = make_cfg()
    model = MLPClassifier(cfg).eval()
    x = torch.randn(B, INPUT_DIM)
    with torch.no_grad():
        logits, feature = model(x)
    assert logits.shape == (B, NUM_CLASSES)
    assert feature.shape == (B, cfg.embed_dim)
    assert torch.isfinite(logits).all()


def test_forward_accepts_optional_label():
    model = MLPClassifier(make_cfg()).eval()
    x = torch.randn(B, INPUT_DIM)
    labels = torch.randint(0, NUM_CLASSES, (B,))
    with torch.no_grad():
        logits, _ = model(x, labels)
    assert logits.shape == (B, NUM_CLASSES)


def test_embedding_matches_forward_feature():
    model = MLPClassifier(make_cfg()).eval()
    x = torch.randn(B, INPUT_DIM)
    with torch.no_grad():
        _, feature = model(x)
        embedding = model.get_embedding(x)
    assert torch.allclose(feature, embedding)


def test_empty_hidden_dims_is_linear_projection():
    cfg = make_cfg(hidden_dims=())
    backbone = MLPBackbone(cfg).eval()
    x = torch.randn(B, INPUT_DIM)
    with torch.no_grad():
        out = backbone(x)
    assert out.shape == (B, cfg.embed_dim)


def test_missing_num_classes_raises():
    with pytest.raises(ValueError):
        MLPClassifier(make_cfg(num_classes=0))


def test_config_rejects_invalid_dropout():
    with pytest.raises(ValueError):
        make_cfg(dropout=1.0)


def test_config_rejects_unknown_activation():
    with pytest.raises(ValueError):
        make_cfg(activation="swish")
