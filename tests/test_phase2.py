import torch
from src.trainer.ppo_loss import compute_m2po_loss
from src.buffer.group_buffer import GroupAwareReplayBuffer
from src.buffer.replay_buffer import VersionedReplayBuffer, VersionedExperience

def test_m2po_loss_is_finite_scalar():
    B, L = 4, 8
    log_probs = torch.randn(B, L)
    old_log_probs = torch.randn(B, L)
    advantages = torch.randn(B, L)
    loss = compute_m2po_loss(log_probs, old_log_probs, advantages)
    assert loss.ndim == 0
    assert torch.isfinite(loss)

def test_group_buffer_emits_when_full():
    buf = GroupAwareReplayBuffer(group_size=4, max_groups=8)
    for i in range(4):
        buf.add_response(
            prompt_id=0,
            response=torch.tensor([i]),
            log_prob=torch.tensor([-0.1]),
            reward=float(i),
            version=0,
        )
    assert buf.ready.qsize() == 1
    group = buf.ready.get_nowait()
    assert group.is_complete is True
    assert len(group.responses) == 4
    assert group.advantages.shape == (4,)

def test_versioned_buffer_evicts_stale():
    buf = VersionedReplayBuffer(max_size=100, max_staleness=5)
    buf.current_version = 20
    buf.push(VersionedExperience(
        prompt_ids=torch.tensor([1]),
        generated_ids=torch.tensor([1, 2]),
        log_probs=torch.tensor([-0.1, -0.2]),
        reward=1.0,
        policy_version=1,
        generation_step=1,
    ))
    assert len(buf.buffer) == 0
