"""
VLLMClient — the real model backend, built on top of the vLLM library.

*** THIS MODULE CANNOT BE UNIT-TESTED IN THIS DEVELOPMENT ENVIRONMENT ***
There is no GPU / vLLM install available here. Everything that depends on
LLMClient (pool building, and later the debate engine + entropy/IGR code)
has already been fully tested against FakeLLMClient — this adapter is the
one remaining piece that must be verified for real, on Kaggle, before any
experiment result from it is trusted.

Run scripts/smoke_test_vllm.py on Kaggle FIRST, and read its output
carefully, before using VLLMClient in scripts/01_build_pools.py or any
later experiment script.
"""

from __future__ import annotations

from typing import Optional

from src.llm.client import GenerationResult, LLMClient
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class VLLMClient(LLMClient):
    def __init__(
        self,
        model_id: str,
        dtype: str = "bfloat16",
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.90,
        tensor_parallel_size: int = 1,
        seed: Optional[int] = None,
    ):
        try:
            from vllm import LLM  # local import: keeps vLLM an optional
            # dependency for anything that only needs FakeLLMClient (tests,
            # CI, or local development off Kaggle).
        except ImportError as exc:
            raise ImportError(
                "vLLM is not installed. On Kaggle: `pip install vllm`. "
                "This dependency is intentionally NOT required for running "
                "the test suite (`python -m pytest`), which uses "
                "FakeLLMClient instead."
            ) from exc

        # vLLM's LLM() constructor requires an int seed, not None — default
        # to 0 rather than silently failing. (Caught during Kaggle smoke
        # testing: T4 sessions passed seed=None through from an unset
        # config value.)
        if seed is None:
            seed = 0

        logger.info(
            "Loading vLLM model '%s' (dtype=%s, max_model_len=%d, "
            "tensor_parallel_size=%d, gpu_memory_utilization=%.2f)...",
            model_id, dtype, max_model_len, tensor_parallel_size, gpu_memory_utilization,
        )
        self.model_id = model_id
        self._llm = LLM(
            model=model_id,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            seed=seed,
        )
        logger.info("vLLM model '%s' loaded.", model_id)

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
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            logprobs=logprobs_topk,
            seed=seed,
        )
        outputs = self._llm.generate([prompt], sampling_params, use_tqdm=False)
        # outputs is a list with one RequestOutput (one prompt in), which
        # itself holds `n` CompletionOutput objects.
        completion_outputs = outputs[0].outputs

        results = []
        for completion in completion_outputs:
            token_logprobs = None
            if logprobs_topk is not None and completion.logprobs is not None:
                # completion.logprobs: list[dict[token_id, Logprob]] per
                # generated token position. Convert to {token_str: logprob}
                # dicts, matching the GenerationResult.token_logprobs contract.
                token_logprobs = [
                    {lp.decoded_token: lp.logprob for lp in position.values()}
                    for position in completion.logprobs
                ]
            results.append(
                GenerationResult(text=completion.text, token_logprobs=token_logprobs)
            )

        if len(results) != n:
            # Defensive check backing the contract every LLMClient consumer
            # relies on (see src/data/pool_generation.py's docstring note).
            raise RuntimeError(
                f"VLLMClient.generate requested n={n} completions but vLLM "
                f"returned {len(results)}. This violates the LLMClient "
                f"contract every calling module assumes — investigate "
                f"before proceeding."
            )
        return results

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
        Batches all of `prompts` into a single vLLM call — vLLM's engine
        schedules distinct prompts together and processes them as one
        batch on the GPU, rather than each call individually paying fixed
        per-call scheduling overhead and running at batch size 1.

        This is what src/agents/debate.py's round-2+ loop uses: every
        agent has a different prompt each round, so this method — not
        `generate`'s n-samples-of-one-prompt batching — is what actually
        parallelizes a debate round across agents. Confirmed as the fix
        for a real multi-day-runtime problem: calling `generate` once per
        agent in a Python loop (the original implementation) left the GPU
        running at batch size 1 for the entire debate loop.
        """
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            n=1,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            logprobs=logprobs_topk,
            seed=seed,
        )
        outputs = self._llm.generate(list(prompts), sampling_params, use_tqdm=False)

        results = []
        for output in outputs:
            completion = output.outputs[0]  # n=1, exactly one completion per prompt
            token_logprobs = None
            if logprobs_topk is not None and completion.logprobs is not None:
                token_logprobs = [
                    {lp.decoded_token: lp.logprob for lp in position.values()}
                    for position in completion.logprobs
                ]
            results.append(
                GenerationResult(text=completion.text, token_logprobs=token_logprobs)
            )

        if len(results) != len(prompts):
            raise RuntimeError(
                f"VLLMClient.generate_batch was given {len(prompts)} prompts "
                f"but vLLM returned {len(results)} results. This violates "
                f"the LLMClient contract every calling module assumes — "
                f"investigate before proceeding."
            )
        return results

    def forced_decode(
        self,
        prompt: str,
        target_text: str,
        logprobs_topk: Optional[int] = None,
    ) -> GenerationResult:
        """
        Teacher-forced decoding: compute the logprobs vLLM assigns to each
        token of `target_text` as a continuation of `prompt`, without
        letting the model generate freely. Implemented via vLLM's
        `prompt_logprobs`: we feed (prompt + target_text) as a single
        prompt with max_tokens=1 (vLLM requires >=1) and prompt_logprobs
        set, then read off the logprobs for the target_text token span.

        NOTE: this is exactly the piece that most needs Kaggle verification
        — confirm via scripts/smoke_test_vllm.py that the returned
        prompt_logprobs actually align with target_text's tokens as
        expected, since off-by-one alignment errors here would silently
        corrupt every downstream entropy/IG/IGR computation (Module 5).
        """
        from vllm import SamplingParams

        full_text = prompt + target_text
        sampling_params = SamplingParams(
            max_tokens=1,
            temperature=0.0,
            prompt_logprobs=logprobs_topk,
        )
        outputs = self._llm.generate([full_text], sampling_params, use_tqdm=False)
        request_output = outputs[0]

        # prompt_logprobs is a list aligned to the *input* tokens of
        # `full_text`; the first entry is always None (no logprob for the
        # very first token). We need only the entries corresponding to the
        # target_text portion, which — token boundaries permitting — is the
        # tail of this list. Exact slicing depends on the tokenizer's
        # handling of the prompt/target boundary; VERIFY on Kaggle.
        prompt_token_ids = request_output.prompt_token_ids
        prompt_logprobs_raw = request_output.prompt_logprobs

        # Re-tokenize just `prompt` to find where target_text's tokens begin.
        tokenizer = self._llm.get_tokenizer()
        prompt_only_ids = tokenizer.encode(prompt)
        boundary = len(prompt_only_ids)

        target_logprobs_raw = prompt_logprobs_raw[boundary:]
        token_logprobs = [
            {lp.decoded_token: lp.logprob for lp in position.values()}
            if position is not None else {}
            for position in target_logprobs_raw
        ]

        return GenerationResult(text=target_text, token_logprobs=token_logprobs)
