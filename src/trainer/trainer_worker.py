# Trainer worker implementation (Ray actor)

import torch
import ray
from typing import List
from ..buffer.replay_buffer import BoundedReplayBuffer, Experience
from .ppo_loss import compute_ppo_loss
from .grpo_loss import compute_grpo_loss

@ray.remote(num_gpus=1)
class TrainerWorker:
    """Ray actor that pulls experiences from a replay buffer and runs PPO/GRPO updates.

    Arguments
    ---------
    model_path: str
        Path to the base policy model (e.g., a HF checkpoint).
    buffer: BoundedReplayBuffer (actor handle)
        Shared replay buffer from which to sample.
    use_grpo: bool, default False
        Whether to train with GRPO (group‑aware) loss.
    """

    def __init__(self, model_path: str, buffer: ray.actor.ActorHandle, use_grpo: bool = False):
        self.buffer = buffer
        self.use_grpo = use_grpo
        # Load a simple lightweight model for demo purposes – replace with actual model loading.
        self.model = torch.nn.Linear(1024, 1024).cuda()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        self.version = 0

    def step(self, batch_size: int = 64):
        """Perform a single training step.

        Returns the loss value (float) or None if not enough data.
        """
        # Pull a batch of experiences
        exps: List[Experience] = ray.get(self.buffer.sample.remote(batch_size))
        if not exps:
            return None

        # Stack tensors
        prompt_ids = torch.stack([e.prompt_ids for e in exps]).cuda()
        gen_ids = torch.stack([e.generated_ids for e in exps]).cuda()
        old_log_probs = torch.stack([e.log_probs for e in exps]).cuda()
        rewards = torch.tensor([e.reward for e in exps], device='cuda')

        # Dummy forward to get current policy log probs (replace with real model forward)
        policy_log_probs = old_log_probs  # placeholder – in practice call model

        # Advantage = reward - baseline (here baseline = 0 for simplicity)
        advantages = rewards.unsqueeze(1).expand_as(policy_log_probs)

        if self.use_grpo:
            # Reshape for group dimension G (placeholder: treat batch as single group)
            policy_log_probs = policy_log_probs.view(1, -1, policy_log_probs.shape[-1])
            old_log_probs = old_log_probs.view(1, -1, old_log_probs.shape[-1])
            advantages = advantages.view(1, -1, advantages.shape[-1])
            loss = compute_grpo_loss(policy_log_probs, old_log_probs, advantages)
        else:
            loss = compute_ppo_loss(policy_log_probs, old_log_probs, advantages)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.version += 1
        return loss.item()

    def get_version(self) -> int:
        return self.version
