# Scheduler and orchestrator utilities

import asyncio
from typing import Any

class Orchestrator:
    """Coordinates rollout, trainer, and weight sync.

    - Maintains a prompt queue (asyncio.Queue).
    - Tracks current policy version via VersionManager.
    - Periodically triggers weight synchronization.
    """

    def __init__(self, version_manager, weight_sync_fn, sync_interval: int = 10):
        self.prompt_queue = asyncio.Queue()
        self.version_manager = version_manager
        self.weight_sync_fn = weight_sync_fn
        self.sync_interval = sync_interval
        self._stop = asyncio.Event()

    async def add_prompt(self, prompt: Any):
        await self.prompt_queue.put(prompt)

    async def run(self):
        """Main orchestration loop.

        1. Every `sync_interval` steps, bump the policy version and call weight sync.
        2. Continuously feed prompts into the queue (could be from an external source).
        """
        step = 0
        while not self._stop.is_set():
            # Example: generate synthetic prompts for demo purposes
            # In practice, populate from dataset or live traffic
            dummy_prompt = {"input_ids": None, "gt_ids": None, "eos_token_id": None}
            await self.add_prompt(dummy_prompt)

            if step % self.sync_interval == 0:
                new_version = self.version_manager.bump_version()
                await self.weight_sync_fn(new_version)
            step += 1
            await asyncio.sleep(0.01)  # tiny pause to avoid tight loop

    def stop(self):
        self._stop.set()
