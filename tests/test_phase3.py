import asyncio
import torch
from src.rollout.vllm_engine import StubEngine
from src.rollout.version_manager import VersionManager


def test_stub_engine_generates():
    async def run():
        engine = StubEngine(vocab_size=100, max_new_tokens=8, eos_token_id=99)
        prompt = torch.tensor([1, 2, 3])
        ids, logprobs = await engine.async_generate(prompt)
        assert ids.ndim == 1
        assert logprobs.ndim == 1
        assert ids.shape[0] == logprobs.shape[0]
        assert ids.shape[0] <= 8
        assert ids.shape[0] >= 1
    asyncio.run(run())


def test_stub_engine_concurrent():
    async def run():
        engine = StubEngine(vocab_size=100, max_new_tokens=4, eos_token_id=99)
        prompts = [torch.tensor([1, 2, 3]) for _ in range(8)]
        results = await asyncio.gather(*[engine.async_generate(p) for p in prompts])
        assert len(results) == 8
        for ids, logprobs in results:
            assert ids.ndim == 1
            assert logprobs.ndim == 1
    asyncio.run(run())


def test_version_manager_bumps():
    vm = VersionManager()
    assert vm.current() == 0
    vm.bump()
    assert vm.current() == 1
    vm.bump()
    vm.bump()
    assert vm.current() == 3
    assert vm.staleness(0) == 3
    assert vm.staleness(3) == 0
