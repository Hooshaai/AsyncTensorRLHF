"""Prompt queue actor for orchestration.

The queue wraps an asyncio.Queue. It provides
`add_prompt` (async) and `get_prompt` methods that can be called
from Ray actors or standalone threads.
"""

import asyncio

try:
    import ray
    ray_remote = ray.remote
except ImportError:
    ray = None
    def ray_remote(cls):
        return cls


@ray_remote
class PromptQueue:
    def __init__(self, maxsize: int = 0):
        self._queue = asyncio.Queue(maxsize=maxsize)

    async def add_prompt(self, prompt: dict):
        await self._queue.put(prompt)
        return True

    async def get_prompt(self):
        prompt = await self._queue.get()
        return prompt

    def size(self):
        return self._queue.qsize()
