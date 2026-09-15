# AsyncEngine that uses the VLLMEngineWrapper

"""AsyncEngine orchestrates async generation and pushes experiences to the replay buffer.

It expects a list of prompt dictionaries where each dictionary contains:
    - ``input_ids``: torch.Tensor (GPU) of the tokenised prompt
    - ``gt_ids``   : torch.Tensor of ground‑truth answer tokens (optional)
    - ``eos_token_id``: int identifier for the EOS token
"""

import asyncio
import uuid
from typing import List
import torch

from .vllm_engine import VLLMEngineWrapper
from ..buffer.replay_buffer import BoundedReplayBuffer, Experience
from ..reward.tensor_native import gpu_reward_simple

class AsyncEngine:
    """Wraps a VLLM async engine and streams generated results.

    The engine is created once per worker and reused for the whole lifetime.
    """

    def __init__(self, model_path: str, buffer: BoundedReplayBuffer, version_manager, **engine_kwargs):
        self.buffer = buffer
        self.version_manager = version_manager
        self.engine = VLLMEngineWrapper(model_path, **engine_kwargs)
        self.id = uuid.uuid4().hex

    async def rollout(self, prompts: List[dict]):
        """Consume a batch of prompts and push experiences as they complete.
        """
        async for result in self.engine.async_generate(prompts):
            # Compute reward on‑GPU without moving data to CPU
            reward_tensor = gpu_reward_simple(
                result.ids.unsqueeze(0),  # batch dim = 1
                result.gt_ids.unsqueeze(0) if result.gt_ids is not None else torch.empty(0, device=result.ids.device),
                result.eos_token_id,
            )
            reward = reward_tensor.item() if reward_tensor.numel() > 0 else 0.0

            exp = Experience(
                prompt_ids=result.input_ids,
                generated_ids=result.ids,
                log_probs=result.logprobs,
                reward=reward,
                version=self.version_manager.get_version(),
            )
            # Non‑blocking push (the buffer handles overflow internally)
            self.buffer.push(exp)

    def set_version(self, new_version: int):
        # The version manager is the source of truth; this is a convenience.
        # Not used directly – orchestrator bumps the version via the manager.
        pass
