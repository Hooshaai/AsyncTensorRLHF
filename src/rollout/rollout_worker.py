# Ray actor for rollout workers

"""RolloutWorker pulls prompts from a shared PromptQueue, generates responses
asynchronously using AsyncEngine, and writes experiences to a ReplayBufferActor.
"""

import ray
import asyncio
from typing import List
from ..rollout.async_engine import AsyncEngine
from ..orchestrator.prompt_queue import PromptQueue
from ..buffer.buffer_actor import ReplayBufferActor
from ..orchestrator.version_manager import VersionManager

@ray.remote(num_gpus=1)
class RolloutWorker:
    def __init__(self, model_path: str, prompt_queue: ray.actor.ActorHandle,
                 replay_buffer: ray.actor.ActorHandle, version_manager: VersionManager,
                 **engine_kwargs):
        self.prompt_queue = prompt_queue
        self.replay_buffer = replay_buffer
        self.version_manager = version_manager
        # local buffer for quick push before sending to actor (optional)
        self.local_buffer = ReplayBufferActor.remote(max_size=5000)
        self.engine = AsyncEngine(model_path, buffer=self.local_buffer, version_manager=self.version_manager, **engine_kwargs)

    async def run(self, batch_size: int = 8):
        """Continuously fetch prompts and perform rollout.

        Args:
            batch_size: number of prompts to fetch per iteration.
        """
        while True:
            # Gather a batch of prompts
            prompts: List[dict] = []
            for _ in range(batch_size):
                prompt = await self.prompt_queue.get_prompt.remote()
                prompts.append(prompt)
            # Run async generation; this yields as soon as each sample finishes
            await self.engine.rollout(prompts)
            # Flush local buffer to shared replay buffer
            # Pull all experiences from local buffer and forward
            while True:
                exps = await self.local_buffer.sample.remote(100)
                if not exps:
                    break
                for exp in exps:
                    await self.replay_buffer.push.remote(exp)
