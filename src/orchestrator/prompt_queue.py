"""Prompt queue actor for orchestration.

The queue is a simple Ray actor wrapping an asyncio.Queue. It provides
`add_prompt` (async) and `get_prompt` (blocking) methods that can be called
from other Ray actors.
"""

import asyncio
import ray

@ray.remote
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
