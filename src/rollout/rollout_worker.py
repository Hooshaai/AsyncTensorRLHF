# Ray actor / standalone worker for rollout workers

"""RolloutWorker pulls prompts from a shared PromptQueue, generates responses
asynchronously using AsyncEngine, and writes experiences to a ReplayBuffer.
"""

from typing import Any, List, Optional

try:
    import ray
    ray_remote = ray.remote(num_gpus=1)
except ImportError:
    ray = None
    def ray_remote(cls):
        return cls

from .async_engine import AsyncEngine
from .version_manager import VersionManager


@ray_remote
class RolloutWorker:
    def __init__(
        self,
        model_path: Optional[str] = None,
        prompt_queue: Optional[Any] = None,
        replay_buffer: Optional[Any] = None,
        buffer: Optional[Any] = None,
        version_manager: Optional[VersionManager] = None,
        engine: Optional[Any] = None,
        **engine_kwargs,
    ):
        self.prompt_queue = prompt_queue
        self.replay_buffer = replay_buffer if replay_buffer is not None else buffer
        self.version_manager = version_manager or VersionManager()
        self.engine = AsyncEngine(
            model_path=model_path,
            buffer=self.replay_buffer,
            version_manager=self.version_manager,
            engine=engine,
            **engine_kwargs,
        )

    async def step(self, prompts: List[dict]):
        """Run rollout on an explicit list of prompts."""
        await self.engine.rollout(prompts)

    async def run(self, batch_size: int = 8):
        """Continuously fetch prompts and perform rollout."""
        while True:
            prompts: List[dict] = []
            for _ in range(batch_size):
                if self.prompt_queue is not None:
                    if hasattr(self.prompt_queue, "get_prompt"):
                        if hasattr(self.prompt_queue.get_prompt, "remote"):
                            prompt = await self.prompt_queue.get_prompt.remote()
                        else:
                            prompt = await self.prompt_queue.get_prompt()
                    else:
                        prompt = await self.prompt_queue.get()
                    prompts.append(prompt)

            if prompts:
                await self.engine.rollout(prompts)
