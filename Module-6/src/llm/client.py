"""
LLM client abstraction.

Why this exists
----------------
Every piece of this project that needs to call a model — pool building,
the debate loop, forced-decoding for entropy/IGR — should depend on this
interface, never on vLLM directly. That gives us two things:

  1. Orchestration logic (branching, retries, prompt construction) can be
     fully unit-tested with FakeLLMClient, with no GPU involved.
  2. The real vLLM adapter (VLLMClient, in vllm_client.py) is the *only*
     piece of the whole project that cannot be verified until we're on
     Kaggle with a GPU attached — everything built on top of this
     interface is already tested by the time we get there.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GenerationResult:
    """One generated completion for one prompt."""

    text: str
    # Per-generated-token top-k logprob distributions, e.g.
    # [{"Yale": -0.1, "Duke": -2.3, ...}, {...}, ...] — one dict per token,
    # in generation order. None if the caller didn't request logprobs.
    # This is what src/entropy/entropy.py (Module 5) will consume.
    token_logprobs: Optional[list] = field(default=None)


class LLMClient(ABC):
    """Minimal interface every model backend (real or fake) must implement."""

    @abstractmethod
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
        """Generate `n` completions for `prompt`. Must return exactly `n` results."""
        raise NotImplementedError

    @abstractmethod
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
        """
        Generate exactly one completion per prompt in `prompts`, batched
        into a single backend call, returned in the same order as `prompts`.

        Exists specifically for the debate loop's round-2+ step: every
        agent has a *different* prompt each round (its own previous
        response + partners' responses differ per agent), so `generate`'s
        n>1-completions-of-one-prompt batching doesn't apply — this is
        n=1-completion-of-many-different-prompts batching instead. Calling
        `generate` once per agent in a loop instead of this method is
        correct but drastically slower: each individual call pays fixed
        per-call scheduling overhead and leaves the GPU underutilized at
        batch size 1, rather than letting the backend batch many distinct
        prompts together in one forward pass.
        """
        raise NotImplementedError

    @abstractmethod
    def forced_decode(
        self,
        prompt: str,
        target_text: str,
        logprobs_topk: Optional[int] = None,
    ) -> GenerationResult:
        """
        Compute the logprobs the model *would* assign to `target_text` if it
        had generated it in response to `prompt`, without actually letting
        the model generate freely (teacher forcing). Required for IG/IGR
        (Module 5) — kept in the base interface now so both the pool
        builder's few-shot step and later DIGRA code share one contract.
        """
        raise NotImplementedError
