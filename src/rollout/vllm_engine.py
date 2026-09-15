# vLLM async engine wrapper

"""A thin wrapper around vLLM's AsyncLLMEngine.

The real implementation would import `vllm` and create an engine that can
asynchronously generate token sequences.  Here we provide a minimal stub that
illustrates the expected API so the rest of the framework can be imported
without pulling the heavy dependency during unit‑tests.

Key methods:
    - ``__init__(model_path, tokenizer, **engine_kwargs)`` – creates the engine.
    - ``async_generate(prompts, sampling_params)`` – returns an async generator
      yielding ``GenerationResult`` objects.  Each result must expose:
        * ``ids`` (torch.Tensor of token ids, shape ``(seq_len,)``)
        * ``logprobs`` (torch.Tensor of log‑probabilities)
        * ``input_ids`` (original prompt token ids)
        * ``gt_ids`` (ground‑truth answer token ids, optional)
        * ``eos_token_id`` (int)
"""

import asyncio
from typing import List, AsyncGenerator
import torch

# In a real implementation you would import the actual vLLM classes:
# from vllm import AsyncLLMEngine, SamplingParams, Request

class GenerationResult:
    """Container for a single generated sequence."""

    def __init__(self, ids, logprobs, input_ids, gt_ids, eos_token_id):
        self.ids = ids  # torch.Tensor (seq_len,)
        self.logprobs = logprobs  # torch.Tensor (seq_len,)
        self.input_ids = input_ids
        self.gt_ids = gt_ids
        self.eos_token_id = eos_token_id


class VLLMEngineWrapper:
    """Placeholder async engine.

    To use the real engine replace the body of ``async_generate`` with calls to
    vLLM's ``engine.generate`` async iterator.
    """

    def __init__(self, model_path: str, tokenizer=None, **engine_kwargs):
        self.model_path = model_path
        self.tokenizer = tokenizer
        # store kwargs for possible real engine creation
        self.engine_kwargs = engine_kwargs
        # self.engine = AsyncLLMEngine(model=model_path, **engine_kwargs)
        # For the stub we do nothing.

    async def async_generate(
        self,
        prompts: List[dict],
        sampling_params: dict = None,
    ) -> AsyncGenerator[GenerationResult, None]:
        """Yield a ``GenerationResult`` for each prompt.

        The stub simply echoes the prompt tokens and appends a dummy EOS token.
        """
        for prompt in prompts:
            # ``prompt`` is expected to contain ``input_ids`` (torch.Tensor)
            input_ids = prompt["input_ids"]
            gt_ids = prompt.get("gt_ids")
            eos_id = prompt.get("eos_token_id", 2)  # assume token id 2 = ``</s>``
            # generate a fake continuation of length 5
            gen_ids = torch.cat([input_ids, torch.arange(100, 105, device=input_ids.device)])
            # fake log probabilities (uniform)
            logprobs = torch.full((gen_ids.shape[0],), -torch.log(torch.tensor(gen_ids.shape[0], dtype=torch.float)), device=gen_ids.device)
            result = GenerationResult(gen_ids, logprobs, input_ids, gt_ids, eos_id)
            # simulate async delay
            await asyncio.sleep(0.001)
            yield result
