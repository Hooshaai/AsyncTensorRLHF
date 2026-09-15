# Group‑aware replay buffer for GRPO

import queue
import threading
from dataclasses import dataclass
from typing import List, Dict
import torch


@dataclass
class GroupBufferEntry:
    prompt_id: int
    responses: List[torch.Tensor]
    log_probs: List[torch.Tensor]
    rewards: List[float]
    policy_version: int
    advantages: torch.Tensor = None
    is_complete: bool = False


class GroupAwareReplayBuffer:
    """Buffers groups of G responses per prompt before exposing them to the trainer.

    - `group_size` defines G.
    - When a group is complete, advantages are computed and the entry is placed on a ready queue.
    - Trainer consumes from the ready queue.
    """

    def __init__(self, group_size: int = 8, max_groups: int = 512):
        self.group_size = group_size
        self.pending: Dict[int, GroupBufferEntry] = {}
        self.ready = queue.Queue(maxsize=max_groups)
        self.lock = threading.Lock()

    def add_response(
        self,
        prompt_id: int,
        response: torch.Tensor,
        log_prob: torch.Tensor,
        reward: float,
        version: int,
    ):
        with self.lock:
            if prompt_id not in self.pending:
                self.pending[prompt_id] = GroupBufferEntry(
                    prompt_id=prompt_id,
                    responses=[],
                    log_probs=[],
                    rewards=[],
                    policy_version=version,
                )
            entry = self.pending[prompt_id]
            entry.responses.append(response)
            entry.log_probs.append(log_prob)
            entry.rewards.append(reward)

            if len(entry.responses) == self.group_size:
                # compute advantages (standardize rewards within group)
                rewards_tensor = torch.tensor(entry.rewards, device=response.device)
                mean_r = rewards_tensor.mean()
                std_r = rewards_tensor.std(unbiased=False) + 1e-8
                entry.advantages = (rewards_tensor - mean_r) / std_r
                entry.is_complete = True
                # push to ready queue (non‑blocking, drop oldest if full)
                try:
                    self.ready.put_nowait(entry)
                except queue.Full:
                    try:
                        self.ready.get_nowait()
                    except queue.Empty:
                        pass
                    self.ready.put_nowait(entry)
                del self.pending[prompt_id]

    def sample_ready(self) -> GroupBufferEntry:
        """Retrieve a completed group; returns None if none available."""
        try:
            return self.ready.get_nowait()
        except queue.Empty:
            return None
