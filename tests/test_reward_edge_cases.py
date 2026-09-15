import pytest
import torch

from src.reward.tensor_native import tensor_native_reward, gpu_reward_simple


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def test_reward_empty_pattern():
    device = get_device()
    gen = torch.tensor([[1, 2, 3, 99, 0]], device=device)
    ans = [torch.empty(0, dtype=torch.long, device=device)]
    r = tensor_native_reward(gen, ans, eos_token_id=99, device=device)
    assert r.shape == (1,)
    assert r[0].item() == 0.0


def test_reward_pattern_longer_than_sequence():
    device = get_device()
    gen = torch.tensor([[1, 2, 99]], device=device)
    ans = [torch.tensor([1, 2, 3, 4, 5], device=device)]
    r = tensor_native_reward(gen, ans, eos_token_id=99, device=device)
    assert r[0].item() == 0.0


def test_reward_no_eos_token_present():
    device = get_device()
    # Sequence never hits EOS token 99; matches anywhere before end of seq
    gen = torch.tensor([[1, 2, 3, 4, 5, 6]], device=device)
    ans = [torch.tensor([3, 4], device=device)]
    r = tensor_native_reward(gen, ans, eos_token_id=99, device=device)
    assert r[0].item() == 1.0


def test_reward_pattern_after_first_eos_is_ignored():
    device = get_device()
    # EOS is at index 2 (token 99). Pattern [5, 6] appears after EOS, so reward must be 0.0
    gen = torch.tensor([[1, 2, 99, 5, 6, 0]], device=device)
    ans = [torch.tensor([5, 6], device=device)]
    r = tensor_native_reward(gen, ans, eos_token_id=99, device=device)
    assert r[0].item() == 0.0


def test_reward_pattern_at_very_beginning():
    device = get_device()
    gen = torch.tensor([[10, 20, 30, 99, 0]], device=device)
    ans = [torch.tensor([10, 20], device=device)]
    r = tensor_native_reward(gen, ans, eos_token_id=99, device=device)
    assert r[0].item() == 1.0


def test_reward_pattern_at_very_end_before_eos():
    device = get_device()
    gen = torch.tensor([[10, 20, 30, 40, 99, 0]], device=device)
    ans = [torch.tensor([30, 40], device=device)]
    r = tensor_native_reward(gen, ans, eos_token_id=99, device=device)
    assert r[0].item() == 1.0


def test_reward_multiple_eos_stops_at_first():
    device = get_device()
    # First EOS is at idx 1. Pattern [30] is after first EOS but before second EOS -> must be 0
    gen = torch.tensor([[10, 99, 30, 99, 0]], device=device)
    ans = [torch.tensor([30], device=device)]
    r = tensor_native_reward(gen, ans, eos_token_id=99, device=device)
    assert r[0].item() == 0.0


def test_reward_batch_heterogeneous_matching():
    device = get_device()
    gen = torch.tensor([
        [1, 2, 3, 4, 99, 0],
        [5, 6, 7, 8, 99, 0],
        [9, 10, 11, 12, 99, 0],
        [13, 14, 15, 16, 99, 0],
    ], device=device)
    ans = [
        torch.tensor([2, 3], device=device),        # Match (1.0)
        torch.tensor([999], device=device),         # No match (0.0)
        torch.tensor([10, 11, 12], device=device),  # Match (1.0)
        torch.tensor([1, 2], device=device),        # No match (0.0)
    ]
    r = tensor_native_reward(gen, ans, eos_token_id=99, device=device)
    assert r.tolist() == [1.0, 0.0, 1.0, 0.0]


def test_gpu_reward_simple_edge_cases():
    device = get_device()
    # Empty ground truth
    gen = torch.tensor([[1, 2, 3, 99]], device=device)
    gt_empty = torch.empty((1, 0), dtype=torch.long, device=device)
    r = gpu_reward_simple(gen, gt_empty, eos_token_id=99)
    assert r[0].item() == 0.0

    # Ground truth longer than generated
    gt_long = torch.tensor([[1, 2, 3, 4, 5]], device=device)
    r = gpu_reward_simple(gen, gt_long, eos_token_id=99)
    assert r[0].item() == 0.0
