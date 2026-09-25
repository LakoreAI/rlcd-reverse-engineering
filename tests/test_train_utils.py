import pytest
import torch
from safetensors.torch import save_file

from src.config import DecisionModelConfig
from src.modules.model import DecisionModel
from src.pipelines.config import TrainingConfig
from src.pipelines.train import (
    _resolve_lr_scheduler,
    build_optimizer,
    load_laya_weights,
)


def test_load_laya_weights_skips_act_head_and_scalar_temperature(
    tmp_path, dummy_encoder
):
    src = DecisionModel(DecisionModelConfig(head_layers=1), encoder=dummy_encoder)
    state = {k: v.clone() for k, v in src.state_dict().items() if k != "temperature"}
    state["act_head.0.weight"] = torch.zeros(4, 4)
    state["temperature"] = torch.ones(3)  # Laya's per-type scalar shape
    path = tmp_path / "model.safetensors"
    save_file(state, str(path))

    from tests.conftest import DummyEncoder

    dst = DecisionModel(DecisionModelConfig(head_layers=1), encoder=DummyEncoder())
    load_laya_weights(dst, str(path))
    for k, v in src.state_dict().items():
        if k != "temperature":
            assert torch.equal(dst.state_dict()[k], v), k


def test_load_laya_weights_rejects_mismatched_checkpoint(tmp_path, dummy_encoder):
    path = tmp_path / "model.safetensors"
    save_file({"something.else": torch.zeros(2)}, str(path))
    model = DecisionModel(DecisionModelConfig(head_layers=1), encoder=dummy_encoder)
    with pytest.raises(RuntimeError):
        load_laya_weights(model, str(path))


def test_build_optimizer_separate_head_lr(dummy_encoder):
    model = DecisionModel(DecisionModelConfig(head_layers=1), encoder=dummy_encoder)
    opt = build_optimizer(model, TrainingConfig(lr=2.5e-5, lr_head=1e-4))
    assert [g["lr"] for g in opt.param_groups] == [2.5e-5, 1e-4]
    n_params = sum(len(g["params"]) for g in opt.param_groups)
    assert n_params == len(list(model.parameters()))


def test_resolve_lr_scheduler_auto_t_max_is_total_steps():
    cfg = TrainingConfig(epochs=4, lr_scheduler={"type": "cosine", "t_max": "auto"})
    assert _resolve_lr_scheduler(cfg, steps_per_epoch=100)["t_max"] == 400
    fixed = TrainingConfig(lr_scheduler={"type": "cosine", "t_max": 7})
    assert _resolve_lr_scheduler(fixed, steps_per_epoch=100)["t_max"] == 7
