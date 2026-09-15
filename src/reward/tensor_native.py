import torch
from typing import List


def gpu_reward_simple(
    generated_ids: torch.Tensor,  # (batch, seq_len) on GPU
    ground_truth_ids: torch.Tensor,  # (batch, gt_len) on GPU
    eos_token_id: int,
) -> torch.Tensor:
    """Simple GPU reward for token‑level subsequence matching.

    Returns a tensor of shape (batch,) with 1.0 for a match and 0.0 otherwise.
    """
    batch_size, seq_len = generated_ids.shape
    rewards = torch.zeros(batch_size, device=generated_ids.device)

    # locate EOS for each sequence
    eos_mask = generated_ids == eos_token_id
    # position of first EOS, or seq_len if none
    first_eos = torch.where(
        eos_mask.any(dim=1),
        eos_mask.int().argmax(dim=1),
        torch.full((batch_size,), seq_len, device=generated_ids.device, dtype=torch.long),
    )

    for i in range(batch_size):
        gen = generated_ids[i, : first_eos[i]]
        gt = ground_truth_ids[i]
        gt_len = gt.shape[0]
        if gt_len > gen.shape[0]:
            continue
        # sliding window comparison
        windows = gen.unfold(0, gt_len, 1)  # (gen_len-gt_len+1, gt_len)
        match = (windows == gt).all(dim=1).any()
        rewards[i] = 1.0 if match else 0.0
    return rewards


def tensor_native_reward(
    generated_ids: torch.Tensor,
    answer_patterns: List[torch.Tensor],
    eos_token_id: int,
    device: str = "cpu",
) -> torch.Tensor:
    """Fully batched tensor‑native reward.

    - Trims each sequence at its first EOS token.
    - For each batch element, checks whether the corresponding answer pattern
      appears as a contiguous subsequence.
    - Returns a (B,) float tensor of rewards (1.0 / 0.0).
    """
    B, L = generated_ids.shape
    device_obj = torch.device(device)
    generated_ids = generated_ids.to(device_obj)
    # Ensure answer patterns are on the same device
    answer_patterns = [p.to(device_obj) for p in answer_patterns]

    rewards = torch.zeros(B, device=device_obj)

    # EOS handling
    eos_mask = generated_ids == eos_token_id
    first_eos = torch.where(
        eos_mask.any(dim=1),
        eos_mask.int().argmax(dim=1),
        torch.full((B,), L, device=device_obj, dtype=torch.long),
    )

    for i in range(B):
        seq = generated_ids[i, : first_eos[i]]
        pattern = answer_patterns[i]
        pat_len = pattern.shape[0]
        if pat_len == 0 or pat_len > seq.shape[0]:
            continue
        windows = seq.unfold(0, pat_len, 1)
        match = (windows == pattern).all(dim=1).any()
        rewards[i] = 1.0 if match else 0.0
    return rewards
