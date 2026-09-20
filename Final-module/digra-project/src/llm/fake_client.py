"""
FakeLLMClient — a deterministic, scriptable stand-in for a real model.

Used exclusively in tests. Two modes:
  1. Queue mode: pre-load exact responses to return in order (for testing
     specific branches, e.g. "first 50 samples are all wrong, forcing the
     few-shot generation path"). Each queued item can be either a plain
     string (text only, token_logprobs=None) or a (text, token_logprobs)
     tuple — the latter lets tests exercise entropy-dependent orchestration
     logic (Module 5's DIGRA loop) through the REAL entropy.py pipeline,
     rather than needing to inject a fake entropy value and bypass it.
  2. Function mode: supply a `response_fn(prompt) -> str` for tests that
     need the response to depend on the prompt content. Function mode
     only produces text (no logprobs) — use queue mode with tuples for
     logprob-dependent tests.
"""

from __future__ import annotations

from collections import deque
from typing import Callable, Optional

from src.llm.client import GenerationResult, LLMClient


class FakeLLMClient(LLMClient):
    def __init__(
        self,
        scripted_responses: Optional[list] = None,
        response_fn: Optional[Callable[[str], str]] = None,
    ):
        if scripted_responses is not None and response_fn is not None:
            raise ValueError("Provide either scripted_responses or response_fn, not both")
        self._queue = deque(scripted_responses or [])
        self._response_fn = response_fn
        self.calls: list[dict] = []  # every call recorded, for test assertions

    def _next_result(self, prompt: str) -> GenerationResult:
        if self._response_fn is not None:
            return GenerationResult(text=self._response_fn(prompt))
        if not self._queue:
            raise RuntimeError(
                "FakeLLMClient scripted_responses exhausted — the code under "
                "test called generate()/generate_batch() more times than the "
                "test anticipated."
            )
        item = self._queue.popleft()
        if isinstance(item, tuple):
            text, token_logprobs = item
            return GenerationResult(text=text, token_logprobs=token_logprobs)
        return GenerationResult(text=item)

    def generate(
        self,
        prompt: str,
        n: int = 1,
        max_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        logprobs_topk: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> list[GenerationResult]:
        self.calls.append({
            "method": "generate", "prompt": prompt, "n": n,
            "max_tokens": max_tokens, "temperature": temperature,
            "top_p": top_p, "top_k": top_k,
        })
        return [self._next_result(prompt) for _ in range(n)]

    def generate_batch(
        self,
        prompts: list[str],
        max_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        logprobs_topk: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> list[GenerationResult]:
        self.calls.append({
            "method": "generate_batch", "prompts": list(prompts), "n": len(prompts),
            "max_tokens": max_tokens, "temperature": temperature,
            "top_p": top_p, "top_k": top_k,
        })
        return [self._next_result(p) for p in prompts]

    def forced_decode(
        self,
        prompt: str,
        target_text: str,
        logprobs_topk: Optional[int] = None,
    ) -> GenerationResult:
        self.calls.append({"method": "forced_decode", "prompt": prompt, "target": target_text})
        token_logprobs = None
        if self._response_fn is None and self._queue:
            item = self._queue.popleft()
            if isinstance(item, tuple):
                _, token_logprobs = item
            # plain-string queue items carry no logprobs for forced_decode;
            # the text itself is ignored here since forced_decode's
            # returned text is always target_text by definition.
        return GenerationResult(text=target_text, token_logprobs=token_logprobs)
