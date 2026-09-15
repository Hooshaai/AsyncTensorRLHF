# Experience replay buffer utilities

import queue
import threading
from dataclasses import dataclass
from typing import List
import torch


@dataclass
class Experience:
    prompt_ids: torch.Tensor
    generated_ids: torch.Tensor
    log_probs: torch.Tensor
    reward: float
    version: int


class BoundedReplayBuffer:
    """Thread‑safe bounded replay buffer.

    - Non‑blocking `push`; if full, discards oldest entry.
    - `sample` returns up to `batch_size` experiences, removing them from the buffer.
    """

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.queue = queue.Queue(maxsize=max_size)
        self.lock = threading.Lock()

    def push(self, exp: Experience):
        with self.lock:
            try:
                self.queue.put_nowait(exp)
            except queue.Full:
                # discard oldest and insert new
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    pass
                self.queue.put_nowait(exp)

    def sample(self, batch_size: int) -> List[Experience]:
        batch: List[Experience] = []
        with self.lock:
            while len(batch) < batch_size:
                try:
                    batch.append(self.queue.get_nowait())
                except queue.Empty:
                    break
        return batch

    def size(self) -> int:
        return self.queue.qsize()


@dataclass
class VersionedExperience:
    prompt_ids: torch.Tensor
    generated_ids: torch.Tensor
    log_probs: torch.Tensor
    reward: float
    policy_version: int
    generation_step: int


class VersionedReplayBuffer:
    """Replay buffer that evicts experiences older than ``max_staleness`` versions.

    - ``max_size``: maximum number of experiences to store.
    - ``max_staleness``: maximum allowed age in policy versions.
      An experience with ``policy_version < current_version - max_staleness`` is dropped.
    - ``current_version``: set externally by the orchestrator before each push.
    """

    def __init__(self, max_size: int = 10000, max_staleness: int = 5):
        self.max_size = max_size
        self.max_staleness = max_staleness
        self.buffer: List[VersionedExperience] = []
        self.current_version: int = 0
        self.lock = threading.Lock()

    def push(self, exp: VersionedExperience) -> None:
        """Push an experience, silently dropping it if it is too stale."""
        with self.lock:
            if exp.policy_version < self.current_version - self.max_staleness:
                return  # stale — drop
            if len(self.buffer) >= self.max_size:
                self.buffer.pop(0)  # evict oldest
            self.buffer.append(exp)

    def sample(self, batch_size: int) -> List[VersionedExperience]:
        """Return up to ``batch_size`` experiences (no removal)."""
        with self.lock:
            return self.buffer[:batch_size]

    def __len__(self) -> int:
        return len(self.buffer)
