# Pure-Python stub rollout engine — no vLLM, no GPU, no Ray.
# Replaces vLLM in Phase 3 for CPU-only testing of the async rollout loop.

import asyncio
import random
from typing import Tuple

import torch


class StubEngine:
    """Minimal async generation engine for testing without vLLM.

    Generates tokens by sampling uniformly from [0, vocab_size) until EOS
    or max_new_tokens is reached. Uses asyncio.sleep(0.001) to force real
    coroutine yielding so that concurrent callers actually interleave.
    """

    def __init__(self, vocab_size: int, max_new_tokens: int, eos_token_id: int):
        self.vocab_size = vocab_size
        self.max_new_tokens = max_new_tokens
        self.eos_token_id = eos_token_id

    async def async_generate(
        self, prompt_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate tokens asynchronously.

        Parameters
        ----------
        prompt_ids : torch.Tensor
            1-D integer tensor (the prompt). Not used for token generation in
            the stub, but accepted to match the real engine's interface.

        Returns
        -------
        generated_ids : torch.Tensor
            1-D long tensor of newly generated token IDs (excludes prompt).
        log_probs : torch.Tensor
            1-D float tensor of per-token log-probabilities (negative floats).
        """
        # Yield once to the event loop so concurrent callers can interleave.
        await asyncio.sleep(0.001)

        generated_ids = []
        log_probs = []

        for _ in range(self.max_new_tokens):
            token_id = random.randint(0, self.vocab_size - 1)
            # Uniform distribution: log(1/vocab_size)
            log_prob = -torch.log(torch.tensor(float(self.vocab_size))).item()
            generated_ids.append(token_id)
            log_probs.append(log_prob)
            if token_id == self.eos_token_id:
                break

        return (
            torch.tensor(generated_ids, dtype=torch.long),
            torch.tensor(log_probs, dtype=torch.float32),
        )
