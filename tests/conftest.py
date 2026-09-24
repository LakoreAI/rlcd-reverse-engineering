"""Shared fixtures for the RLCD test suite.

`DummyTokenizer` / `DummyEncoder` stand in for a real HF tokenizer/encoder so
unit tests stay offline and fast — they only need to satisfy the small
surface `src.data.build_sequence` and `src.modules.model.DecisionModel`
actually call, not the full `transformers` API.
"""

from types import SimpleNamespace

import pytest
from torch import nn


class DummyTokenizer:
    mask_token = "[MASK]"
    mask_token_id = 1
    cls_token_id = 2
    sep_token_id = 3
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=True, truncation=False, max_length=None):
        ids = [10 + (hash(word) % 50) for word in text.split()] or [10]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids}


class DummyEncoder(nn.Module):
    """A tiny embedding-only stand-in for an `AutoModel` encoder."""

    def __init__(self, hidden_size: int = 16, vocab_size: int = 128):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embed = nn.Embedding(vocab_size, hidden_size)

    def forward(self, input_ids, attention_mask=None):
        ids = input_ids.clamp(max=self.embed.num_embeddings - 1)
        return SimpleNamespace(last_hidden_state=self.embed(ids))


@pytest.fixture
def dummy_tokenizer() -> DummyTokenizer:
    return DummyTokenizer()


@pytest.fixture
def dummy_encoder() -> DummyEncoder:
    return DummyEncoder()
