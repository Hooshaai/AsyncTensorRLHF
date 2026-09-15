"""Comprehensive Benchmark Suite for AsyncTensorRLHF.

Benchmarks:
1. In-VRAM Tensor-Native Reward vs. CPU-Roundtrip Baseline across multiple batch sizes.
2. PPO, M2PO, and GRPO loss & backward throughput (samples/sec & tokens/sec).
3. Replay buffer throughput (push/sample ops/sec under concurrency).
4. Staleness robustness benchmark (PPO vs. M2PO gradient variance under policy drift).
"""

import os
import pathlib
import sys
import time
import torch

# Ensure project root in sys.path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.reward.tensor_native import tensor_native_reward
from src.trainer.ppo_loss import compute_ppo_loss, compute_m2po_loss
from src.trainer.grpo_loss import compute_grpo_loss
from src.buffer.replay_buffer import BoundedReplayBuffer, VersionedReplayBuffer, Experience, VersionedExperience
from src.buffer.group_buffer import GroupAwareReplayBuffer


def run_reward_benchmark(device):
    print("\n" + "=" * 65)
    print("1. TENSOR-NATIVE REWARD vs. CPU-ROUNDTRIP BASELINE BENCHMARK")
    print("=" * 65)
    print(f"{'Batch Size':<12} | {'Seq Len':<10} | {'Tensor-Native (ms)':<20} | {'CPU-Decode (ms)':<18} | {'Speedup':<10}")
    print("-" * 65)

    batch_sizes = [16, 32, 64, 128, 256]
    seq_len = 256
    pat_len = 4

    results = []

    for B in batch_sizes:
        gen_ids = torch.randint(0, 5000, (B, seq_len), device=device)
        patterns = [torch.randint(0, 5000, (pat_len,), device=device) for _ in range(B)]

        # Warmup
        _ = tensor_native_reward(gen_ids, patterns, eos_token_id=99, device=device)
        if device == "cuda":
            torch.cuda.synchronize()

        # Benchmark Tensor-Native (In-VRAM)
        iters = 20
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = tensor_native_reward(gen_ids, patterns, eos_token_id=99, device=device)
        if device == "cuda":
            torch.cuda.synchronize()
        t_native = (time.perf_counter() - t0) / iters * 1000

        # Benchmark CPU-Decode baseline (simulate copy to CPU and string/list search)
        t0 = time.perf_counter()
        for _ in range(iters):
            cpu_ids = gen_ids.cpu().tolist()
            cpu_pats = [p.cpu().tolist() for p in patterns]
            rewards = []
            for b in range(B):
                row = cpu_ids[b]
                pat = cpu_pats[b]
                match = any(row[i:i+len(pat)] == pat for i in range(len(row) - len(pat) + 1))
                rewards.append(1.0 if match else 0.0)
            _ = torch.tensor(rewards, device=device)
        if device == "cuda":
            torch.cuda.synchronize()
        t_cpu = (time.perf_counter() - t0) / iters * 1000

        speedup = t_cpu / max(t_native, 1e-4)
        print(f"{B:<12} | {seq_len:<10} | {t_native:<20.2f} | {t_cpu:<18.2f} | {speedup:<10.1f}x")
        results.append((B, seq_len, t_native, t_cpu, speedup))

    return results


def run_loss_benchmark(device):
    print("\n" + "=" * 65)
    print("2. POLICY LOSS THROUGHPUT (PPO vs. M2PO vs. GRPO)")
    print("=" * 65)
    print(f"{'Loss Type':<12} | {'Batch Size':<12} | {'Forward+Backward (ms)':<25} | {'Tokens/sec':<15}")
    print("-" * 65)

    B = 64
    L = 256
    iters = 30

    # PPO
    p_lp = torch.randn(B, L, device=device, requires_grad=True)
    o_lp = torch.randn(B, L, device=device)
    adv = torch.randn(B, L, device=device)

    # Warmup
    loss = compute_ppo_loss(p_lp, o_lp, adv)
    loss.backward()

    t0 = time.perf_counter()
    for _ in range(iters):
        p_lp.grad = None
        loss = compute_ppo_loss(p_lp, o_lp, adv)
        loss.backward()
    if device == "cuda":
        torch.cuda.synchronize()
    t_ppo = (time.perf_counter() - t0) / iters * 1000
    tok_ppo = (B * L) / (t_ppo / 1000)
    print(f"{'PPO':<12} | {B:<12} | {t_ppo:<25.2f} | {tok_ppo:<15.0f}")

    # M2PO
    p_lp.grad = None
    t0 = time.perf_counter()
    for _ in range(iters):
        p_lp.grad = None
        loss = compute_m2po_loss(p_lp, o_lp, adv, m2_threshold=2.0)
        loss.backward()
    if device == "cuda":
        torch.cuda.synchronize()
    t_m2po = (time.perf_counter() - t0) / iters * 1000
    tok_m2po = (B * L) / (t_m2po / 1000)
    print(f"{'M2PO':<12} | {B:<12} | {t_m2po:<25.2f} | {tok_m2po:<15.0f}")

    # GRPO
    G = 4
    B_grpo = B // G
    p_grp = torch.randn(B_grpo, G, L, device=device, requires_grad=True)
    o_grp = torch.randn(B_grpo, G, L, device=device)
    adv_grp = torch.randn(B_grpo, G, L, device=device)

    t0 = time.perf_counter()
    for _ in range(iters):
        p_grp.grad = None
        loss = compute_grpo_loss(p_grp, o_grp, adv_grp)
        loss.backward()
    if device == "cuda":
        torch.cuda.synchronize()
    t_grpo = (time.perf_counter() - t0) / iters * 1000
    tok_grpo = (B * L) / (t_grpo / 1000)
    print(f"{'GRPO':<12} | {B:<12} | {t_grpo:<25.2f} | {tok_grpo:<15.0f}")


def run_buffer_benchmark():
    print("\n" + "=" * 65)
    print("3. REPLAY BUFFER THROUGHPUT BENCHMARK")
    print("=" * 65)

    buf = BoundedReplayBuffer(max_size=50000)
    N = 20000
    exp = Experience(
        prompt_ids=torch.tensor([1, 2, 3]),
        generated_ids=torch.tensor([4, 5, 6, 7]),
        log_probs=torch.tensor([-0.1, -0.2, -0.1, -0.3]),
        reward=1.0,
        version=0,
    )

    t0 = time.perf_counter()
    for _ in range(N):
        buf.push(exp)
    t_push = time.perf_counter() - t0
    push_rate = N / t_push

    t0 = time.perf_counter()
    sampled = 0
    while sampled < N:
        batch = buf.sample(64)
        if not batch:
            break
        sampled += len(batch)
    t_sample = time.perf_counter() - t0
    sample_rate = sampled / t_sample

    print(f"Push Throughput:   {push_rate:,.0f} ops/sec ({N} items pushed in {t_push*1000:.1f} ms)")
    print(f"Sample Throughput: {sample_rate:,.0f} ops/sec ({sampled} items sampled in {t_sample*1000:.1f} ms)")


def run_staleness_benchmark(device):
    print("\n" + "=" * 65)
    print("4. ASYNCHRONOUS STALENESS ROBUSTNESS (PPO vs. M2PO)")
    print("=" * 65)
    print(f"{'Staleness tau':<15} | {'PPO Grad Norm':<18} | {'M2PO Grad Norm':<18} | {'Variance Reduction':<20}")
    print("-" * 65)

    B, L = 32, 64
    staleness_levels = [0, 1, 2, 3, 5, 8]

    for tau in staleness_levels:
        # Drift std proportional to staleness tau
        drift = 0.15 * tau
        old_lp = torch.randn(B, L, device=device)
        policy_lp_ppo = (old_lp + torch.randn(B, L, device=device) * drift).clone().detach().requires_grad_(True)
        policy_lp_m2po = policy_lp_ppo.clone().detach().requires_grad_(True)
        adv = torch.randn(B, L, device=device)

        loss_p = compute_ppo_loss(policy_lp_ppo, old_lp, adv)
        loss_p.backward()
        p_norm = policy_lp_ppo.grad.norm().item()

        loss_m = compute_m2po_loss(policy_lp_m2po, old_lp, adv, m2_threshold=2.0)
        loss_m.backward()
        m_norm = policy_lp_m2po.grad.norm().item()

        red = ((p_norm - m_norm) / max(p_norm, 1e-6)) * 100
        print(f"tau = {tau:<10} | {p_norm:<18.4f} | {m_norm:<18.4f} | {red:<20.1f}%")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 65)
    print("AsyncTensorRLHF Comprehensive System Benchmark")
    print(f"Hardware Platform: {device.upper()}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} (Capability: {torch.cuda.get_device_capability(0)})")
    print("=" * 65)

    run_reward_benchmark(device)
    run_loss_benchmark(device)
    run_buffer_benchmark()
    run_staleness_benchmark(device)
    print("\n" + "=" * 65)
    print("ALL COMPREHENSIVE BENCHMARKS COMPLETED SUCCESSFULLY")
    print("=" * 65)


if __name__ == "__main__":
    main()
