# Trainer worker implementation (Ray actor or standalone worker)

from typing import Any, List, Optional
import torch

try:
    import ray
    ray_remote = ray.remote(num_gpus=1)
except ImportError:
    ray = None
    def ray_remote(cls):
        return cls

from ..buffer.replay_buffer import BoundedReplayBuffer, Experience
from .ppo_loss import compute_ppo_loss, compute_m2po_loss
from .grpo_loss import compute_grpo_loss


@ray_remote
class TrainerWorker:
    """Ray actor or standalone worker that samples experiences and runs PPO/M2PO/GRPO updates.

    Arguments
    ---------
    model_path: str or torch.nn.Module, optional
        Path to model or model module.
    buffer: Any
        Shared replay buffer or actor handle.
    use_grpo: bool, default False
        Whether to train with GRPO loss.
    use_m2po: bool, default False
        Whether to train with M2PO loss.
    device: str, optional
        Device to use ('cuda' or 'cpu').
    """

    def __init__(
        self,
        model_path: Optional[Any] = None,
        buffer: Optional[Any] = None,
        use_grpo: bool = False,
        use_m2po: bool = False,
        device: Optional[str] = None,
        lr: float = 1e-4,
    ):
        self.buffer = buffer
        self.use_grpo = use_grpo
        self.use_m2po = use_m2po
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        if isinstance(model_path, torch.nn.Module):
            self.model = model_path.to(self.device)
        else:
            self.model = torch.nn.Sequential(
                torch.nn.Linear(128, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 128),
            ).to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.version = 0

    def step(self, batch_size: int = 64) -> Optional[float]:
        """Perform a single training step."""
        if self.buffer is None:
            return None

        # Sample from buffer (actor handle or local instance)
        if hasattr(self.buffer, "sample"):
            if hasattr(self.buffer.sample, "remote") and ray is not None:
                exps: List[Experience] = ray.get(self.buffer.sample.remote(batch_size))
            else:
                exps: List[Experience] = self.buffer.sample(batch_size)
        else:
            return None

        if not exps:
            return None

        # Pad or stack tensors
        max_prompt = max(e.prompt_ids.shape[-1] for e in exps)
        max_gen = max(e.generated_ids.shape[-1] for e in exps)

        padded_old_logprobs = []
        padded_advantages = []

        for e in exps:
            lp = e.log_probs.to(self.device)
            if lp.shape[-1] < max_gen:
                lp = torch.cat([lp, torch.zeros(max_gen - lp.shape[-1], device=self.device)])
            padded_old_logprobs.append(lp)
            r = float(e.reward)
            padded_advantages.append(torch.full((max_gen,), r, device=self.device))

        old_log_probs = torch.stack(padded_old_logprobs)
        advantages = torch.stack(padded_advantages)

        # Compute simulated policy log probs with small perturbation for optimization
        dummy_in = torch.randn(len(exps), 128, device=self.device)
        rep = self.model(dummy_in)
        policy_log_probs = old_log_probs + 0.01 * rep.mean(dim=-1, keepdim=True)

        if self.use_grpo:
            loss = compute_grpo_loss(
                policy_log_probs.unsqueeze(0),
                old_log_probs.unsqueeze(0),
                advantages.unsqueeze(0),
            )
        elif self.use_m2po:
            loss = compute_m2po_loss(policy_log_probs, old_log_probs, advantages)
        else:
            loss = compute_ppo_loss(policy_log_probs, old_log_probs, advantages)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.version += 1
        return float(loss.item())

    def get_version(self) -> int:
        return self.version
