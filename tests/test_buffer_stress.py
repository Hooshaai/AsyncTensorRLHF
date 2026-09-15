import concurrent.futures
import queue
import torch
import pytest

from src.buffer.replay_buffer import (
    BoundedReplayBuffer,
    Experience,
    VersionedReplayBuffer,
    VersionedExperience,
)
from src.buffer.group_buffer import GroupAwareReplayBuffer
from src.buffer.buffer_actor import ReplayBufferActor


def test_bounded_buffer_overflow_discards_oldest():
    buf = BoundedReplayBuffer(max_size=3)
    for i in range(5):
        buf.push(Experience(
            prompt_ids=torch.tensor([i]),
            generated_ids=torch.tensor([i]),
            log_probs=torch.tensor([0.0]),
            reward=float(i),
            version=i,
        ))
    assert buf.size() == 3
    sampled = buf.sample(3)
    # The oldest entries 0 and 1 were evicted; remaining should be 2, 3, 4
    rewards = [e.reward for e in sampled]
    assert rewards == [2.0, 3.0, 4.0]


def test_bounded_buffer_multithreaded_stress():
    buf = BoundedReplayBuffer(max_size=500)
    num_threads = 8
    items_per_thread = 50

    def producer(thread_id):
        for j in range(items_per_thread):
            buf.push(Experience(
                prompt_ids=torch.tensor([thread_id]),
                generated_ids=torch.tensor([j]),
                log_probs=torch.tensor([-0.1]),
                reward=float(thread_id * 100 + j),
                version=0,
            ))

    def consumer():
        collected = []
        for _ in range(items_per_thread // 2):
            batch = buf.sample(4)
            collected.extend(batch)
        return collected

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads * 2) as executor:
        producer_futures = [executor.submit(producer, t) for t in range(num_threads)]
        consumer_futures = [executor.submit(consumer) for _ in range(num_threads)]
        concurrent.futures.wait(producer_futures + consumer_futures)

    # Buffer remains consistent and doesn't crash or deadlock
    assert buf.size() >= 0
    assert buf.size() <= 500


def test_versioned_buffer_exact_boundaries():
    buf = VersionedReplayBuffer(max_size=10, max_staleness=4)
    buf.current_version = 10  # cutoff is 10 - 4 = 6. Allowed: 6, 7, 8, 9, 10

    # 1. Stale: version 5 < 6 -> dropped
    buf.push(VersionedExperience(
        prompt_ids=torch.tensor([1]),
        generated_ids=torch.tensor([1]),
        log_probs=torch.tensor([-0.1]),
        reward=1.0,
        policy_version=5,
        generation_step=0,
    ))
    assert len(buf) == 0

    # 2. Boundary: version 6 == 6 -> kept
    buf.push(VersionedExperience(
        prompt_ids=torch.tensor([1]),
        generated_ids=torch.tensor([1]),
        log_probs=torch.tensor([-0.1]),
        reward=1.0,
        policy_version=6,
        generation_step=0,
    ))
    assert len(buf) == 1

    # 3. Fresh: version 10 == 10 -> kept
    buf.push(VersionedExperience(
        prompt_ids=torch.tensor([1]),
        generated_ids=torch.tensor([1]),
        log_probs=torch.tensor([-0.1]),
        reward=1.0,
        policy_version=10,
        generation_step=0,
    ))
    assert len(buf) == 2


def test_group_buffer_zero_variance_rewards():
    # If all responses in a group receive identical rewards (e.g. all 1.0 or all 0.0),
    # standard deviation is 0. Advantages must not produce NaN or Inf due to division by zero.
    buf = GroupAwareReplayBuffer(group_size=4, max_groups=4)
    for i in range(4):
        buf.add_response(
            prompt_id=999,
            response=torch.tensor([i]),
            log_prob=torch.tensor([-0.2]),
            reward=1.0,  # identical rewards
            version=0,
        )

    assert buf.ready.qsize() == 1
    group = buf.ready.get_nowait()
    assert torch.isfinite(group.advantages).all()
    # All advantages should be 0.0 when rewards are identical
    assert torch.allclose(group.advantages, torch.zeros(4))


def test_group_buffer_interleaved_prompts():
    buf = GroupAwareReplayBuffer(group_size=3, max_groups=8)

    # Prompt A: sample 0, 1
    buf.add_response(10, torch.tensor([1]), torch.tensor([-0.1]), 0.0, 0)
    buf.add_response(10, torch.tensor([2]), torch.tensor([-0.1]), 1.0, 0)
    assert buf.ready.qsize() == 0

    # Prompt B: sample 0
    buf.add_response(20, torch.tensor([10]), torch.tensor([-0.1]), 0.5, 0)
    assert buf.ready.qsize() == 0

    # Prompt A: sample 2 -> Prompt A finishes and emits
    buf.add_response(10, torch.tensor([3]), torch.tensor([-0.1]), 2.0, 0)
    assert buf.ready.qsize() == 1
    group_a = buf.ready.get_nowait()
    assert group_a.prompt_id == 10
    assert len(group_a.responses) == 3

    # Prompt B: sample 1, 2 -> Prompt B finishes and emits
    buf.add_response(20, torch.tensor([20]), torch.tensor([-0.1]), 0.5, 0)
    buf.add_response(20, torch.tensor([30]), torch.tensor([-0.1]), 0.5, 0)
    assert buf.ready.qsize() == 1
    group_b = buf.ready.get_nowait()
    assert group_b.prompt_id == 20


def test_buffer_actor_standalone():
    actor = ReplayBufferActor(max_size=10)
    exp = Experience(
        prompt_ids=torch.tensor([1]),
        generated_ids=torch.tensor([2]),
        log_probs=torch.tensor([-0.1]),
        reward=1.0,
        version=1,
    )
    assert actor.push(exp) is True
    assert actor.size() == 1
    sampled = actor.sample(5)
    assert len(sampled) == 1
    assert actor.size() == 0
