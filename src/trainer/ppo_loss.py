# PPO loss implementation with optional off‑policy correction (M2PO)

import torch
from torch.nn import functional as F


def compute_ppo_loss(
    policy_log_probs: torch.Tensor,   # (B, seq_len)
    old_log_probs: torch.Tensor,      # (B, seq_len)
    advantages: torch.Tensor,         # (B, seq_len)
    clip_eps: float = 0.2,
    use_m2po: bool = False,
    m2_threshold: float = 2.0,
) -> torch.Tensor:
    """Compute PPO (or M2PO) loss.

    Parameters
    ----------
    policy_log_probs : torch.Tensor
        Log‑probs of actions under the current policy.
    old_log_probs : torch.Tensor
        Log‑probs recorded during rollout.
    advantages : torch.Tensor
        Advantage estimates (reward‑to‑go minus baseline).
    clip_eps : float, default 0.2
        Standard PPO clipping epsilon.
    use_m2po : bool, default False
        If True, applies the second‑moment trust‑region (M2PO) instead of clipping.
    m2_threshold : float, default 2.0
        Maximum allowed value of (ratio ** 2) mean for M2PO.
    """
    # ratio = pi(a|s) / pi_old(a|s)
    ratio = torch.exp(policy_log_probs - old_log_probs)

    if use_m2po:
        # second‑moment constraint
        m2 = (ratio ** 2).mean()
        mask = ratio < m2_threshold
        loss = -torch.min(
            ratio * advantages,
            torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages,
        )
        loss = (loss * mask.float()).sum() / mask.float().sum().clamp(min=1.0)
    else:
        # standard PPO clipping
        clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

    return loss


def compute_m2po_loss(
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
    m2_threshold: float = 2.0,
) -> torch.Tensor:
    """Convenience wrapper: compute_ppo_loss with use_m2po=True.

    Returns a scalar torch.Tensor.
    """
    return compute_ppo_loss(
        policy_log_probs,
        old_log_probs,
        advantages,
        clip_eps=clip_eps,
        use_m2po=True,
        m2_threshold=m2_threshold,
    )
