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
