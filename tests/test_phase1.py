import torch
from src.reward.tensor_native import tensor_native_reward
from src.buffer.replay_buffer import BoundedReplayBuffer, Experience
from src.trainer.ppo_loss import compute_ppo_loss


def test_reward_match():
    generated_ids = torch.tensor([
        [1, 2, 3, 4, 99, 0],
        [1, 5, 6, 7, 99, 0],
    ])
    answer_patterns = [torch.tensor([2, 3, 4]), torch.tensor([8, 9])]
    rewards = tensor_native_reward(
        generated_ids=generated_ids,
        answer_patterns=answer_patterns,
        eos_token_id=99,
        device="cpu",
    )
    assert rewards.shape == (2,)
    assert rewards[0].item() == 1.0
    assert rewards[1].item() == 0.0


def test_replay_buffer():
    buf = BoundedReplayBuffer(max_size=10)
    for i in range(5):
        exp = Experience(
            prompt_ids=torch.tensor([i]),
            generated_ids=torch.tensor([i, i + 1]),
            log_probs=torch.tensor([-0.1, -0.2]),
            reward=float(i),
            version=0,
        )
        buf.push(exp)
    assert buf.size() == 5
    batch = buf.sample(3)
    assert len(batch) == 3


def test_ppo_loss_scalar():
    B, L = 4, 8
    log_probs = torch.randn(B, L)
    old_log_probs = torch.randn(B, L)
    advantages = torch.randn(B, L)
    loss = compute_ppo_loss(log_probs, old_log_probs, advantages)
    assert loss.ndim == 0
