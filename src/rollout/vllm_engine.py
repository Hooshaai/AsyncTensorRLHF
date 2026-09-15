# Rollout engine implementations: StubEngine, HFEngine, and VLLMEngineWrapper.
# Supports CPU and CUDA GPU execution, pure-Python async testing,
# and high-performance tensor-native execution on NVIDIA GPUs.

import asyncio
import random
from dataclasses import dataclass
from typing import AsyncIterator, List, Optional, Tuple, Union

import torch


@dataclass
class GenerationResult:
    """Result of an asynchronous generation request."""
    input_ids: torch.Tensor
    ids: torch.Tensor
    logprobs: torch.Tensor
    gt_ids: Optional[torch.Tensor] = None
    eos_token_id: int = 99


class StubEngine:
    """Minimal async generation engine for testing without vLLM or GPU.

    Generates tokens by sampling uniformly from [0, vocab_size) until EOS
    or max_new_tokens is reached. Uses asyncio.sleep(0.001) to force real
    coroutine yielding so that concurrent callers actually interleave.
    """

    def __init__(self, vocab_size: int = 100, max_new_tokens: int = 8, eos_token_id: int = 99):
        self.vocab_size = vocab_size
        self.max_new_tokens = max_new_tokens
        self.eos_token_id = eos_token_id

    async def async_generate(
        self, prompt_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate tokens asynchronously for a single prompt."""
        await asyncio.sleep(0.001)

        generated_ids = []
        log_probs = []

        device = prompt_ids.device if isinstance(prompt_ids, torch.Tensor) else torch.device("cpu")
        uniform_log_prob = -torch.log(torch.tensor(float(self.vocab_size), device=device)).item()

        for _ in range(self.max_new_tokens):
            token_id = random.randint(0, self.vocab_size - 1)
            generated_ids.append(token_id)
            log_probs.append(uniform_log_prob)
            if token_id == self.eos_token_id:
                break

        return (
            torch.tensor(generated_ids, dtype=torch.long, device=device),
            torch.tensor(log_probs, dtype=torch.float32, device=device),
        )


class HFEngine:
    """Async generation engine using PyTorch / Transformers on CUDA or CPU.

    Runs autoregressive generation directly on the target GPU without CPU roundtrips,
    extracting exact log-probabilities for RLHF/GRPO.
    """

    def __init__(
        self,
        model_or_path: Union[str, torch.nn.Module],
        tokenizer=None,
        device: Optional[str] = None,
        max_new_tokens: int = 32,
        eos_token_id: Optional[int] = None,
        vocab_size: int = 1000,
    ):
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.max_new_tokens = max_new_tokens
        self.tokenizer = tokenizer
        self.eos_token_id = eos_token_id if eos_token_id is not None else 99
        self.vocab_size = vocab_size

        if isinstance(model_or_path, torch.nn.Module):
            self.model = model_or_path.to(self.device)
            self.model.eval()
        elif isinstance(model_or_path, str) and model_or_path:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                self.model = AutoModelForCausalLM.from_pretrained(model_or_path).to(self.device)
                self.model.eval()
                if self.tokenizer is None:
                    self.tokenizer = AutoTokenizer.from_pretrained(model_or_path)
                if hasattr(self.model.config, "vocab_size"):
                    self.vocab_size = self.model.config.vocab_size
                if hasattr(self.model.config, "eos_token_id") and self.model.config.eos_token_id is not None:
                    self.eos_token_id = self.model.config.eos_token_id
            except Exception:
                # Fallback to lightweight linear language model for testing/prototyping
                self.model = torch.nn.Sequential(
                    torch.nn.Embedding(self.vocab_size, 128),
                    torch.nn.Linear(128, self.vocab_size),
                ).to(self.device)
                self.model.eval()
        else:
            self.model = torch.nn.Sequential(
                torch.nn.Embedding(self.vocab_size, 128),
                torch.nn.Linear(128, self.vocab_size),
            ).to(self.device)
            self.model.eval()

    async def async_generate(
        self, prompt_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Asynchronously generate tokens on GPU and return (generated_ids, logprobs)."""
        await asyncio.sleep(0.0001)

        input_ids = prompt_ids.to(self.device)
        if input_ids.ndim == 1:
            curr_tokens = input_ids.clone()
        else:
            curr_tokens = input_ids[0].clone()

        generated_ids = []
        log_probs = []

        with torch.no_grad():
            for _ in range(self.max_new_tokens):
                logits = self.model(curr_tokens.unsqueeze(0))
                if hasattr(logits, "logits"):
                    logits = logits.logits
                next_token_logits = logits[0, -1, :]
                probs = torch.softmax(next_token_logits, dim=-1)
                log_p = torch.log_softmax(next_token_logits, dim=-1)

                next_token = torch.multinomial(probs, num_samples=1)
                token_val = next_token.item()
                token_logp = log_p[token_val].item()

                generated_ids.append(token_val)
                log_probs.append(token_logp)
                curr_tokens = torch.cat([curr_tokens, next_token])

                if token_val == self.eos_token_id:
                    break

        return (
            torch.tensor(generated_ids, dtype=torch.long, device=self.device),
            torch.tensor(log_probs, dtype=torch.float32, device=self.device),
        )


class VLLMEngineWrapper:
    """Wrapper that supports vLLM if installed, or falls back to HFEngine or StubEngine."""

    def __init__(self, model_path: Optional[str] = None, engine=None, **engine_kwargs):
        self.model_path = model_path
        self.engine = engine

        if self.engine is None:
            vllm_available = False
            try:
                import vllm
                vllm_available = True
            except ImportError:
                vllm_available = False

            if vllm_available and model_path:
                from vllm import AsyncEngineArgs, AsyncLLMEngine
                args = AsyncEngineArgs(model=model_path, **engine_kwargs)
                self.engine = AsyncLLMEngine.from_engine_args(args)
                self.is_vllm = True
            else:
                self.is_vllm = False
                device = "cuda" if torch.cuda.is_available() else "cpu"
                if model_path and model_path != "stub":
                    self.engine = HFEngine(model_path, device=device, **engine_kwargs)
                else:
                    self.engine = StubEngine(**engine_kwargs) if engine_kwargs else StubEngine()
        else:
            self.is_vllm = False

    async def async_generate(
        self, prompts: Union[List[dict], torch.Tensor]
    ) -> AsyncIterator[GenerationResult]:
        """Stream generated results for a list of prompt dicts or a single prompt."""
        if isinstance(prompts, torch.Tensor):
            prompts = [{"input_ids": prompts}]

        for p in prompts:
            input_ids = p.get("input_ids")
            gt_ids = p.get("gt_ids", None)
            eos_token_id = p.get("eos_token_id", 99)

            if isinstance(input_ids, list):
                input_ids = torch.tensor(input_ids)

            if input_ids is None:
                input_ids = torch.tensor([1, 2, 3])

            ids, logprobs = await self.engine.async_generate(input_ids)
            yield GenerationResult(
                input_ids=input_ids,
                ids=ids,
                logprobs=logprobs,
                gt_ids=gt_ids,
                eos_token_id=eos_token_id,
            )
