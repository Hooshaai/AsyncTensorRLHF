import asyncio
import threading
import torch
import pytest

from src.orchestrator.prompt_queue import PromptQueue
from src.orchestrator.scheduler import Orchestrator
from src.rollout.version_manager import VersionManager
from src.rollout.vllm_engine import StubEngine, HFEngine
from src.rollout.rollout_worker import RolloutWorker
from src.trainer.trainer_worker import TrainerWorker
from src.buffer.replay_buffer import BoundedReplayBuffer, Experience


def test_version_manager_concurrency():
    vm = VersionManager()
    num_threads = 20
    increments_per_thread = 50

    def worker():
        for _ in range(increments_per_thread):
            vm.bump()

    threads = [threading.Thread(target=worker) for _ in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert vm.current() == num_threads * increments_per_thread
    assert vm.staleness(0) == num_threads * increments_per_thread


def test_prompt_queue_async_operations():
    async def run():
        pq = PromptQueue(maxsize=5)
        assert pq.size() == 0

        p1 = {"input_ids": torch.tensor([1, 2, 3])}
        p2 = {"input_ids": torch.tensor([4, 5, 6])}

        await pq.add_prompt(p1)
        await pq.add_prompt(p2)
        assert pq.size() == 2

        out1 = await pq.get_prompt()
        assert torch.equal(out1["input_ids"], p1["input_ids"])
        assert pq.size() == 1

        out2 = await pq.get_prompt()
        assert torch.equal(out2["input_ids"], p2["input_ids"])
        assert pq.size() == 0

    asyncio.run(run())


def test_rollout_worker_step_execution():
    buf = BoundedReplayBuffer(max_size=10)
    vm = VersionManager()
    stub = StubEngine(vocab_size=50, max_new_tokens=4, eos_token_id=99)
    worker = RolloutWorker(buffer=buf, version_manager=vm, engine=stub)

    prompts = [
        {"input_ids": torch.tensor([1, 2]), "gt_ids": torch.tensor([3]), "eos_token_id": 99},
        {"input_ids": torch.tensor([4, 5]), "gt_ids": torch.tensor([5]), "eos_token_id": 99},
        {"input_ids": torch.tensor([6, 7]), "gt_ids": torch.tensor([7]), "eos_token_id": 99},
    ]

    async def run():
        await worker.step(prompts)
        assert buf.size() == 3

    asyncio.run(run())


def test_multi_step_trainer_optimizer_updates():
    buf = BoundedReplayBuffer(max_size=20)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for i in range(10):
        buf.push(Experience(
            prompt_ids=torch.tensor([1, 2], device=device),
            generated_ids=torch.tensor([i, i+1, i+2], device=device),
            log_probs=torch.tensor([-0.1, -0.2, -0.3], device=device),
            reward=float(i % 2),
            version=0,
        ))

    trainer = TrainerWorker(buffer=buf, device=device, lr=1e-3)
    initial_version = trainer.get_version()
    assert initial_version == 0

    losses = []
    for _ in range(3):
        loss = trainer.step(batch_size=4)
        assert loss is not None
        assert isinstance(loss, float)
        losses.append(loss)

    assert trainer.get_version() == 3
    assert len(losses) == 3
    for l in losses:
        assert isinstance(l, float)


def test_orchestrator_weight_sync_callback():
    vm = VersionManager()
    buf = BoundedReplayBuffer(max_size=20)
    stub = StubEngine(vocab_size=32, max_new_tokens=4, eos_token_id=99)
    rollout = RolloutWorker(buffer=buf, version_manager=vm, engine=stub)
    trainer = TrainerWorker(buffer=buf)

    sync_history = []
    async def async_sync(version):
        sync_history.append(version)

    orch = Orchestrator(
        version_manager=vm,
        weight_sync_fn=async_sync,
        sync_interval=2,
        rollout_worker=rollout,
        trainer_worker=trainer,
        buffer=buf,
    )

    async def run():
        losses = await orch.run(max_steps=4)
        assert len(losses) > 0
        assert len(sync_history) >= 2

    asyncio.run(run())
