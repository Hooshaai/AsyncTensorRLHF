import asyncio
import torch
import pytest

from src.reward.tensor_native import tensor_native_reward, gpu_reward_simple
from src.buffer.replay_buffer import BoundedReplayBuffer, VersionedReplayBuffer, Experience, VersionedExperience
from src.buffer.group_buffer import GroupAwareReplayBuffer
from src.trainer.ppo_loss import compute_ppo_loss, compute_m2po_loss
from src.trainer.grpo_loss import compute_grpo_loss
from src.trainer.trainer_worker import TrainerWorker
from src.rollout.vllm_engine import StubEngine, HFEngine, VLLMEngineWrapper
from src.rollout.async_engine import AsyncEngine
from src.rollout.rollout_worker import RolloutWorker
from src.rollout.version_manager import VersionManager
from src.orchestrator.scheduler import Orchestrator


def get_test_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def test_device_native_reward():
    device = get_test_device()
    gen = torch.tensor([[10, 20, 30, 40, 99, 0], [10, 50, 60, 70, 99, 0]], device=device)
    ans = [torch.tensor([20, 30, 40], device=device), torch.tensor([88, 99], device=device)]

    rewards = tensor_native_reward(gen, ans, eos_token_id=99, device=device)
    assert rewards.device.type == device
    assert rewards.shape == (2,)
    assert rewards[0].item() == 1.0
    assert rewards[1].item() == 0.0


def test_gpu_reward_simple():
    device = get_test_device()
    gen = torch.tensor([[5, 6, 7, 8, 99]], device=device)
    gt = torch.tensor([[6, 7, 8]], device=device)

    r = gpu_reward_simple(gen, gt, eos_token_id=99)
    assert r.device.type == device
    assert r[0].item() == 1.0


def test_all_losses_on_device():
    device = get_test_device()
    B, L = 4, 16
    p_lp = torch.randn(B, L, device=device, requires_grad=True)
    o_lp = torch.randn(B, L, device=device)
    adv = torch.randn(B, L, device=device)

    # PPO
    l_ppo = compute_ppo_loss(p_lp, o_lp, adv)
    assert l_ppo.device.type == device
    assert torch.isfinite(l_ppo)
    l_ppo.backward()

    # M2PO
    p_lp2 = torch.randn(B, L, device=device, requires_grad=True)
    l_m2po = compute_m2po_loss(p_lp2, o_lp, adv)
    assert l_m2po.device.type == device
    assert torch.isfinite(l_m2po)
    l_m2po.backward()

    # GRPO
    G = 4
    p_grp = torch.randn(2, G, L, device=device, requires_grad=True)
    o_grp = torch.randn(2, G, L, device=device)
    adv_grp = torch.randn(2, G, L, device=device)
    l_grpo = compute_grpo_loss(p_grp, o_grp, adv_grp)
    assert l_grpo.device.type == device
    assert torch.isfinite(l_grpo)
    l_grpo.backward()


def test_hf_engine_autoregressive():
    device = get_test_device()
    engine = HFEngine(model_or_path=None, device=device, max_new_tokens=6, vocab_size=50, eos_token_id=49)
    prompt = torch.tensor([1, 2, 3], device=device)

    async def run():
        ids, logprobs = await engine.async_generate(prompt)
        assert ids.device.type == device
        assert logprobs.device.type == device
        assert ids.ndim == 1
        assert logprobs.ndim == 1
        assert ids.shape[0] == logprobs.shape[0]

    asyncio.run(run())


def test_async_engine_rollout_and_buffer_push():
    device = get_test_device()
    buf = BoundedReplayBuffer(max_size=20)
    vm = VersionManager()
    stub = StubEngine(vocab_size=64, max_new_tokens=6, eos_token_id=99)
    engine = AsyncEngine(buffer=buf, version_manager=vm, engine=stub)

    prompts = [
        {"input_ids": torch.tensor([1, 2], device=device), "gt_ids": torch.tensor([10], device=device), "eos_token_id": 99},
        {"input_ids": torch.tensor([3, 4], device=device), "gt_ids": torch.tensor([20], device=device), "eos_token_id": 99},
    ]

    async def run():
        await engine.rollout(prompts)
        assert buf.size() == 2
        exps = buf.sample(2)
        assert len(exps) == 2
        assert exps[0].reward in (0.0, 1.0)

    asyncio.run(run())


def test_trainer_worker_step():
    device = get_test_device()
    buf = BoundedReplayBuffer(max_size=10)
    for i in range(4):
        buf.push(Experience(
            prompt_ids=torch.tensor([1, 2], device=device),
            generated_ids=torch.tensor([3, 4, 5], device=device),
            log_probs=torch.tensor([-0.2, -0.3, -0.1], device=device),
            reward=1.0,
            version=0,
        ))

    trainer = TrainerWorker(buffer=buf, device=device)
    loss = trainer.step(batch_size=4)
    assert loss is not None
    assert isinstance(loss, float)
    assert trainer.get_version() == 1


def test_orchestrator_closed_loop():
    device = get_test_device()
    buf = BoundedReplayBuffer(max_size=20)
    vm = VersionManager()
    stub = StubEngine(vocab_size=32, max_new_tokens=4, eos_token_id=99)
    rollout_worker = RolloutWorker(buffer=buf, version_manager=vm, engine=stub)
    trainer_worker = TrainerWorker(buffer=buf, device=device)

    sync_calls = []
    def sync_fn(v):
        sync_calls.append(v)

    orch = Orchestrator(
        version_manager=vm,
        weight_sync_fn=sync_fn,
        sync_interval=2,
        rollout_worker=rollout_worker,
        trainer_worker=trainer_worker,
        buffer=buf,
    )

    async def run():
        losses = await orch.run(max_steps=5)
        assert len(losses) > 0
        assert len(sync_calls) > 0

    asyncio.run(run())
