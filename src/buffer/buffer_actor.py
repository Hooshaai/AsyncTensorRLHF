# Buffer actor supporting Ray or standalone in-memory execution

from typing import Any, List

try:
    import ray
    ray_remote = ray.remote
except ImportError:
    ray = None
    def ray_remote(cls):
        return cls

from ..buffer.replay_buffer import BoundedReplayBuffer, Experience


@ray_remote
class ReplayBufferActor:
    def __init__(self, max_size: int = 10000):
        self.buffer = BoundedReplayBuffer(max_size=max_size)

    def push(self, exp: Experience) -> bool:
        self.buffer.push(exp)
        return True

    def sample(self, batch_size: int) -> List[Experience]:
        return self.buffer.sample(batch_size)

    def size(self) -> int:
        return self.buffer.size()
