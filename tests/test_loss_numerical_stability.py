import pytest
import torch

from src.trainer.ppo_loss import compute_ppo_loss, compute_m2po_loss
from src.trainer.grpo_loss import compute_grpo_loss


def test_ppo_identical_policies_ratio_one():
    # When policy_log_probs == old_log_probs, ratio is exactly 1.0 everywhere.
    # Loss should be -mean(advantages)
    B, L = 4, 10
    lp = torch.randn(B, L)
    adv = torch.randn(B, L)
    loss = compute_ppo_loss(lp, lp, adv)
    expected = -adv.mean()
    assert torch.isclose(loss, expected, atol=1e-5)


def test_ppo_clipping_bounds():
    # Extreme policy divergence: ratio is huge (exp(10) >> 1.2)
    B, L = 2, 4
    old_lp = torch.zeros(B, L)
    new_lp = torch.full((B, L), 10.0)  # ratio = e^10 ~ 22026
    adv = torch.ones(B, L)  # positive advantage

    loss = compute_ppo_loss(new_lp, old_lp, adv, clip_eps=0.2)
    # Since adv > 0 and ratio > 1+eps, clipped term (1+eps)*adv = 1.2 is smaller than ratio*adv
    # Loss should be -1.2
    assert torch.isclose(loss, torch.tensor(-1.2), atol=1e-5)


def test_m2po_threshold_masking():
    # If ratio^2 exceeds m2_threshold, the M2PO mask drops it or bounds it
    B, L = 2, 4
    old_lp = torch.zeros(B, L)
    # Sequence with high divergence
    new_lp = torch.full((B, L), 2.0)  # ratio = e^2 ~ 7.389, ratio^2 ~ 54.6 > m2_threshold=2.0
    adv = torch.ones(B, L)

    loss_m2po = compute_m2po_loss(new_lp, old_lp, adv, m2_threshold=2.0)
    assert torch.isfinite(loss_m2po)


def test_grpo_loss_gradient_flow():
    B, G, L = 2, 4, 8
    policy_lp = torch.randn(B, G, L, requires_grad=True)
    old_lp = torch.randn(B, G, L)
    advantages = torch.randn(B, G, L)

    loss = compute_grpo_loss(policy_lp, old_lp, advantages, clip_eps=0.2)
    loss.backward()

    assert policy_lp.grad is not None
    assert torch.isfinite(policy_lp.grad).all()
    assert policy_lp.grad.shape == policy_lp.shape


def test_loss_backward_passes_with_zero_advantages():
    B, L = 4, 8
    policy_lp = torch.randn(B, L, requires_grad=True)
    old_lp = torch.randn(B, L)
    zero_adv = torch.zeros(B, L)

    # PPO
    l_ppo = compute_ppo_loss(policy_lp, old_lp, zero_adv)
    l_ppo.backward()
    assert torch.allclose(policy_lp.grad, torch.zeros_like(policy_lp.grad))
    policy_lp.grad.zero_()

    # M2PO
    l_m2po = compute_m2po_loss(policy_lp, old_lp, zero_adv)
    l_m2po.backward()
    assert torch.allclose(policy_lp.grad, torch.zeros_like(policy_lp.grad))
