# Ray actor that wraps the BoundedReplayBuffer

import ray
from typing import List
from ..buffer.replay_buffer import BoundedReplayBuffer, Experience

@ray.remote
class ReplayBufferActor:
    def __init__(self, max_size: int = 10000):
        self.buffer = BoundedReplayBuffer(max_size=max_size)

    def push(self, exp: Experience):
        self.buffer.push(exp)
        return True

    def sample(self, batch_size: int) -> List[Experience]:
        return self.buffer.sample(batch_size)

    def size(self) -> int:
        return self.buffer.size()
