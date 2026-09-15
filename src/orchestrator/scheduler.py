# Scheduler and orchestrator utilities for asynchronous RLHF

import asyncio
from typing import Any, Callable, List, Optional
import torch

from ..rollout.version_manager import VersionManager


class Orchestrator:
    """Coordinates rollout, buffer, trainer, and weight synchronization.

    - Maintains a prompt queue.
    - Tracks current policy version via VersionManager.
    - Periodically triggers weight synchronization and training updates.
    """

    def __init__(
        self,
        version_manager: Optional[VersionManager] = None,
        weight_sync_fn: Optional[Callable] = None,
        sync_interval: int = 10,
        rollout_worker: Optional[Any] = None,
        trainer_worker: Optional[Any] = None,
        buffer: Optional[Any] = None,
    ):
        self.prompt_queue = asyncio.Queue()
        self.version_manager = version_manager or VersionManager()
        self.weight_sync_fn = weight_sync_fn
        self.sync_interval = sync_interval
        self.rollout_worker = rollout_worker
        self.trainer_worker = trainer_worker
        self.buffer = buffer
        self._stop = asyncio.Event()

    async def add_prompt(self, prompt: Any):
        await self.prompt_queue.put(prompt)

    async def step_training(self, batch_size: int = 16) -> Optional[float]:
        """Execute one training step if a trainer worker is configured."""
        if self.trainer_worker is None:
            return None
        if hasattr(self.trainer_worker, "step"):
            if hasattr(self.trainer_worker.step, "remote"):
                return await self.trainer_worker.step.remote(batch_size)
            return self.trainer_worker.step(batch_size)
        return None

    async def run(self, max_steps: Optional[int] = None):
        """Main orchestration loop."""
        step = 0
        losses: List[float] = []

        while not self._stop.is_set():
            if max_steps is not None and step >= max_steps:
                break

            # If rollout worker is configured, feed a prompt batch
            if self.rollout_worker is not None:
                prompts = [{"input_ids": torch.tensor([1, 2, 3]), "gt_ids": torch.tensor([2, 3]), "eos_token_id": 99}]
                if hasattr(self.rollout_worker, "step"):
                    await self.rollout_worker.step(prompts)

            # Perform a trainer step
            loss = await self.step_training()
            if loss is not None:
                losses.append(loss)

            # Check if sync interval reached
            if step % self.sync_interval == 0:
                new_version = self.version_manager.bump()
                if self.weight_sync_fn is not None:
                    res = self.weight_sync_fn(new_version)
                    if asyncio.iscoroutine(res):
                        await res

            step += 1
            await asyncio.sleep(0.001)

        return losses

    def stop(self):
        self._stop.set()
