import pytest
import torch

from src.config import DecisionModelConfig
from src.modules.model import DecisionModel

B, L, K = 4, 20, 5


def make_batch(vocab_size=100):
    ids = torch.randint(0, vocab_size, (B, L))
    attention_mask = torch.ones(B, L, dtype=torch.long)
    marker_pos = torch.randint(0, L, (B, K))
    marker_mask = torch.ones(B, K, dtype=torch.bool)
    marker_mask[:, -1] = False  # last option padded, for one row
    qtype = torch.zeros(B, dtype=torch.long)
    return ids, attention_mask, marker_pos, marker_mask, qtype


def test_forward_shapes(dummy_encoder):
    cfg = DecisionModelConfig(head_layers=1)
    model = DecisionModel(cfg, encoder=dummy_encoder).eval()
    ids, attention_mask, marker_pos, marker_mask, qtype = make_batch()
    with torch.no_grad():
        logits = model(ids, attention_mask, marker_pos, marker_mask, qtype)
    assert logits.shape == (B, K)
    assert torch.isfinite(logits).all()


def test_masked_positions_get_pad_logit(dummy_encoder):
    cfg = DecisionModelConfig(head_layers=1)
    model = DecisionModel(cfg, encoder=dummy_encoder).eval()
    ids, attention_mask, marker_pos, marker_mask, qtype = make_batch()
    with torch.no_grad():
        logits = model(ids, attention_mask, marker_pos, marker_mask, qtype)
    assert (logits[~marker_mask] == -1e4).all()
    assert (logits[marker_mask] != -1e4).all()


def test_type_embedding_changes_logits(dummy_encoder):
    cfg = DecisionModelConfig(head_layers=1, num_qtypes=3)
    model = DecisionModel(cfg, encoder=dummy_encoder).eval()
    ids, attention_mask, marker_pos, marker_mask, _qtype = make_batch()
    qtype_a = torch.zeros(B, dtype=torch.long)
    qtype_b = torch.full((B,), 2, dtype=torch.long)
    with torch.no_grad():
        logits_a = model(ids, attention_mask, marker_pos, marker_mask, qtype_a)
        logits_b = model(ids, attention_mask, marker_pos, marker_mask, qtype_b)
    assert not torch.allclose(logits_a, logits_b)


def test_temperature_buffer_shape(dummy_encoder):
    cfg = DecisionModelConfig(head_layers=1, num_qtypes=3, k_buckets=(2, 5, 10))
    model = DecisionModel(cfg, encoder=dummy_encoder)
    assert model.temperature.shape == (3, 4)
    assert torch.allclose(model.temperature, torch.ones(3, 4))


def test_config_rejects_invalid_head_layers():
    with pytest.raises(ValueError):
        DecisionModelConfig(head_layers=0)


def test_config_rejects_invalid_dropout():
    with pytest.raises(ValueError):
        DecisionModelConfig(head_dropout=1.0)


def test_config_rejects_head_max_len_over_max_len():
    with pytest.raises(ValueError):
        DecisionModelConfig(max_len=32, head_max_len=64)
