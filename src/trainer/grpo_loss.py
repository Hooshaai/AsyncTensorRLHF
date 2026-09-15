# GRPO loss implementation (group‑aware advantage normalization)

import torch
from torch.nn import functional as F


def compute_grpo_loss(
    policy_log_probs: torch.Tensor,   # (B, G, seq_len) where G is group size
    old_log_probs: torch.Tensor,      # same shape as above
    advantages: torch.Tensor,         # (B, G, seq_len) – already normalized per group
    clip_eps: float = 0.2,
) -> torch.Tensor:
    """Compute GRPO loss.

    The loss is applied per‑token, per‑response within each group.
    Advantages are assumed to have been computed by standardizing rewards
    across the G responses of a prompt (see `GroupAwareReplayBuffer`).
    """
    ratio = torch.exp(policy_log_probs - old_log_probs)
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    loss = -torch.min(ratio * advantages, clipped * advantages).mean()
    return loss
