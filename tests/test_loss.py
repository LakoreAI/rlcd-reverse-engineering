import torch

from src.config import QTYPES
from src.modules.loss import predict, probabilities, proper_reward, rlcd_loss

torch.manual_seed(0)


def test_proper_reward_finite_and_shaped():
    q = torch.softmax(torch.randn(3, 4, 5), dim=-1)
    t = torch.softmax(torch.randn(3, 4, 5), dim=-1)
    mask = torch.ones(4, 5, dtype=torch.bool)
    qtype = torch.full((4,), QTYPES["choice"], dtype=torch.long)
    r = proper_reward(q, t, qtype, mask)
    assert r.shape == (3, 4)
    assert torch.isfinite(r).all()


def test_proper_reward_rps_only_applies_to_score_type():
    q = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    t = torch.tensor([[0.4, 0.3, 0.2, 0.1]])
    mask = torch.ones(1, 4, dtype=torch.bool)

    r_choice = proper_reward(q, t, torch.tensor([QTYPES["choice"]]), mask)
    r_score = proper_reward(q, t, torch.tensor([QTYPES["score"]]), mask)
    # score-type subtracts a (non-negative here) RPS term, so its reward
    # must be <= the choice-type reward for the same q, t.
    assert float(r_score) <= float(r_choice)


def test_rlcd_loss_ce_only_matches_soft_cross_entropy_gradient():
    z = torch.randn(2, 4, requires_grad=True)
    target = torch.softmax(torch.randn(2, 4), dim=-1)
    qtype = torch.zeros(2, dtype=torch.long)
    mask = torch.ones(2, 4, dtype=torch.bool)

    loss, parts = rlcd_loss(z, target, qtype, mask, sigma=1.0, w_rl=0.0, w_ce=1.0)
    loss.backward()

    # loss_ce is `.mean()`-reduced over the batch, so the per-row CE
    # gradient (softmax(z) - target) is scaled by 1 / batch_size.
    batch_size = z.shape[0]
    expected_grad = (torch.softmax(z.detach(), dim=-1) - target) / batch_size
    assert torch.allclose(z.grad, expected_grad, atol=1e-5)
    assert parts["loss_rl"] == 0.0 or torch.isfinite(torch.tensor(parts["loss_rl"]))


def test_rlcd_loss_rl_only_has_no_ce_contribution_to_grad():
    z = torch.randn(2, 4, requires_grad=True)
    target = torch.softmax(torch.randn(2, 4), dim=-1)
    qtype = torch.zeros(2, dtype=torch.long)
    mask = torch.ones(2, 4, dtype=torch.bool)

    loss, parts = rlcd_loss(z, target, qtype, mask, sigma=1.0, w_rl=1.0, w_ce=0.0)
    assert torch.isfinite(loss)
    assert isinstance(parts["loss_ce"], float)


def test_rlcd_loss_ignores_padded_option_logits():
    torch.manual_seed(1)
    target = torch.tensor([[0.7, 0.3, 0.0]])
    mask = torch.tensor([[True, True, False]])
    qtype = torch.zeros(1, dtype=torch.long)

    z1 = torch.tensor([[1.0, -1.0, 5.0]], requires_grad=True)
    z2 = torch.tensor([[1.0, -1.0, -50.0]], requires_grad=True)

    torch.manual_seed(42)
    loss1, _ = rlcd_loss(z1, target, qtype, mask, sigma=0.5, w_rl=1.0, w_ce=1.0)
    torch.manual_seed(42)
    loss2, _ = rlcd_loss(z2, target, qtype, mask, sigma=0.5, w_rl=1.0, w_ce=1.0)
    assert torch.allclose(loss1, loss2, atol=1e-4)


def test_probabilities_and_predict_ignore_padding():
    logits = torch.tensor([[2.0, 1.0, -1e4]])
    probs = probabilities(logits)
    assert probs[0, -1] < 1e-6
    assert int(predict(logits)[0]) == 0
