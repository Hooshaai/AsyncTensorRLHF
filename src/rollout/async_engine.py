# AsyncEngine that uses VLLMEngineWrapper / HFEngine / StubEngine

"""AsyncEngine orchestrates async generation and pushes experiences to the replay buffer.

It expects a list of prompt dictionaries where each dictionary contains:
    - ``input_ids``: torch.Tensor of the tokenised prompt
    - ``gt_ids``   : torch.Tensor of ground‑truth answer tokens (optional)
    - ``eos_token_id``: int identifier for the EOS token
"""

import asyncio
import uuid
from typing import Any, List, Optional
import torch

from .vllm_engine import VLLMEngineWrapper
from ..buffer.replay_buffer import BoundedReplayBuffer, Experience, VersionedExperience, VersionedReplayBuffer
from ..reward.tensor_native import gpu_reward_simple


class AsyncEngine:
    """Wraps an async engine and streams generated results to replay buffers.

    The engine is created once per worker and reused for the whole lifetime.
    Supports CPU, CUDA GPU, and distributed execution.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        buffer: Optional[Any] = None,
        version_manager: Optional[Any] = None,
        engine: Optional[Any] = None,
        **engine_kwargs,
    ):
        self.buffer = buffer
        self.version_manager = version_manager
        if engine is not None:
            self.engine = VLLMEngineWrapper(engine=engine, **engine_kwargs)
        else:
            self.engine = VLLMEngineWrapper(model_path=model_path, **engine_kwargs)
        self.id = uuid.uuid4().hex

    def _get_version(self) -> int:
        if self.version_manager is None:
            return 0
        if hasattr(self.version_manager, "current"):
            return self.version_manager.current()
        if hasattr(self.version_manager, "get_version"):
            return self.version_manager.get_version()
        if isinstance(self.version_manager, int):
            return self.version_manager
        return 0

    async def rollout(self, prompts: List[dict]):
        """Consume a batch of prompts and push experiences as they complete."""
        async for result in self.engine.async_generate(prompts):
            device = result.ids.device if isinstance(result.ids, torch.Tensor) else torch.device("cpu")
            gt_ids = (
                result.gt_ids.to(device)
                if result.gt_ids is not None
                else torch.empty(0, device=device, dtype=torch.long)
            )

            # Compute reward on device (CUDA GPU or CPU) without host roundtrips
            reward_tensor = gpu_reward_simple(
                result.ids.unsqueeze(0),
                gt_ids.unsqueeze(0) if gt_ids.numel() > 0 else gt_ids,
                result.eos_token_id,
            )
            reward = reward_tensor.item() if reward_tensor.numel() > 0 else 0.0

            version = self._get_version()

            if self.buffer is not None:
                if isinstance(self.buffer, VersionedReplayBuffer):
                    exp = VersionedExperience(
                        prompt_ids=result.input_ids,
                        generated_ids=result.ids,
                        log_probs=result.logprobs,
                        reward=reward,
                        policy_version=version,
                        generation_step=0,
                    )
                else:
                    exp = Experience(
                        prompt_ids=result.input_ids,
                        generated_ids=result.ids,
                        log_probs=result.logprobs,
                        reward=reward,
                        version=version,
                    )

                # Push to buffer (actor handle or local object)
                if hasattr(self.buffer, "push"):
                    if hasattr(self.buffer.push, "remote"):
                        # Ray actor call
                        self.buffer.push.remote(exp)
                    else:
                        self.buffer.push(exp)

    def set_version(self, new_version: int):
        pass
