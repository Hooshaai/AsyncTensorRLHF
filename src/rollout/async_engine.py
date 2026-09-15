import asyncio
import uuid
from typing import List

from ..reward.tensor_native import gpu_reward_simple
from ..buffer.replay_buffer import BoundedReplayBuffer, Experience

class AsyncEngine:
    """Async inference wrapper around vLLM or SGLang.

    The engine yields `GenerationResult` objects as soon as a sample reaches EOS.
    """

    def __init__(self, engine, buffer: BoundedReplayBuffer, version: int = 0):
        self.engine = engine  # underlying vLLM/SGLang engine instance
        self.buffer = buffer
        self.version = version
        self.id = uuid.uuid4().hex

    async def rollout(self, prompts: List[dict]):
        """Run async generation for a batch of prompts.

        Each prompt dict should contain:
            - "input_ids": torch.Tensor on GPU
            - "gt_ids": torch.Tensor of ground‑truth answer tokens (optional)
            - "eos_token_id": int
        """
        async for result in self.engine.async_generate(prompts):
            # result has .ids, .logprobs, .eos_token_id
            reward = gpu_reward_simple(result.ids, result.gt_ids, result.eos_token_id)
            exp = Experience(
                prompt_ids=result.input_ids,
                generated_ids=result.ids,
                log_probs=result.logprobs,
                reward=reward.item(),
                version=self.version,
            )
            # non‑blocking push
            self.buffer.push(exp)

    def set_version(self, new_version: int):
        self.version = new_version
