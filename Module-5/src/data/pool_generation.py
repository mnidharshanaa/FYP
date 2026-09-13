"""
Pool-building orchestration (Appendix B-3, full procedure).

This is the thin layer that actually calls a model, but it depends only on
the LLMClient interface (src/llm/client.py) — so every branch of the
step-2/step-3 logic is exercised in tests/test_pool_generation.py using
FakeLLMClient, with zero GPU involved. Only the concrete VLLMClient
implementation (src/llm/vllm_client.py) remains unverified until Kaggle.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from src.data.pool_builder import select_correct_pool
from src.data.prompts import build_guided_correct_prompt, build_neutral_prompt
from src.llm.client import LLMClient
from src.metrics.grading import extract_final_answer, is_correct
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class PoolResult:
    question_id: str
    correct_texts: list         # final correct-response pool, length <= n_all
    incorrect_texts: list       # FARM's seeded logical-appeal passages, used as-is
    difficulty: float           # n1 / n_attempts, from the initial neutral sampling
    n_correct_sampled: int      # n1
    n_attempts: int             # total neutral samples drawn (50 in the paper)
    n_generated_via_fewshot: int
    fully_satisfied: bool       # True if we reached n_all correct responses


def build_pool_for_question(
    llm: LLMClient,
    question_id: str,
    question: str,
    gold_answer: str,
    alternatives: list,
    logical_appeals: list,
    n_all: int = 5,
    n_attempts: int = 50,
    seed: int = 0,
    max_tokens: int = 300,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 50,
) -> PoolResult:
    """
    Full Appendix B-3 procedure for one question:
      step 1: sample the neutral prompt n_attempts times.
      step 2: if >= n_all came back correct, randomly retain n_all.
      step 3: otherwise, keep all correct ones, then generate exactly the
              shortfall via a single guided (few-shot + gold-answer) batch,
              accepted as-is per the paper's design (guided generation is
              trusted, not re-filtered).

    `max_tokens`/`temperature`/`top_p`/`top_k` should be sourced from
    configs/base.yaml's `generation` section by the caller — NOT left at
    this function's defaults in a real run, since the guided prompt
    (step 3) embeds prior generations as few-shot examples, and an
    oversized max_tokens here compounds directly into that prompt's
    length (this is exactly what caused a real context-length failure on
    Kaggle when max_tokens defaulted to 1024 unwired from config; see
    src/data/prompts.py's max_examples cap for the second layer of
    defense against the same failure mode).

    Note on design: `llm.generate(n=k)` is contractually guaranteed to
    return exactly k completions (see src/llm/client.py), so step 3 always
    exactly fills the shortfall in one batch — there is no retry loop here,
    because one would never have anything to do. `fully_satisfied` is kept
    as a defensive check only, in case a future LLMClient implementation
    ever violates that contract.
    """
    rng = random.Random(seed)

    neutral_prompt = build_neutral_prompt(question)
    step1_results = llm.generate(
        neutral_prompt, n=n_attempts, seed=seed,
        max_tokens=max_tokens, temperature=temperature, top_p=top_p, top_k=top_k,
    )
    sampled_texts = [r.text for r in step1_results]

    selection = select_correct_pool(sampled_texts, gold_answer, alternatives, n_all, rng)
    correct_pool = list(selection.correct_texts)
    n_generated_via_fewshot = 0

    if selection.n_needed > 0:
        guided_prompt = build_guided_correct_prompt(
            question=question, gold_answer=gold_answer, correct_examples=correct_pool,
        )
        extra_results = llm.generate(
            guided_prompt, n=selection.n_needed, seed=seed + 1000,
            max_tokens=max_tokens, temperature=temperature, top_p=top_p, top_k=top_k,
        )
        extra_texts = [r.text for r in extra_results]

        n_extra_verified_correct = sum(
            1 for t in extra_texts
            if is_correct(extract_final_answer(t), gold_answer, alternatives)
        )
        if n_extra_verified_correct < len(extra_texts):
            logger.warning(
                "Question %s: guided-generation batch produced %d/%d responses "
                "that don't verify as correct on re-grading; accepting all per "
                "Appendix B-3 step 3 (guided generation is trusted, not "
                "re-filtered) but flagging for visibility.",
                question_id, n_extra_verified_correct, len(extra_texts),
            )

        correct_pool.extend(extra_texts)
        n_generated_via_fewshot = len(extra_texts)

    fully_satisfied = len(correct_pool) >= n_all
    if not fully_satisfied:
        logger.error(
            "Question %s: pool has only %d/%d correct responses after step 3 — "
            "this should be unreachable given LLMClient's contract (generate(n=k) "
            "must return exactly k results). Investigate the LLMClient "
            "implementation in use.",
            question_id, len(correct_pool), n_all,
        )

    correct_pool = correct_pool[:n_all]

    return PoolResult(
        question_id=question_id,
        correct_texts=correct_pool,
        incorrect_texts=list(logical_appeals),
        difficulty=selection.difficulty,
        n_correct_sampled=selection.n_correct_sampled,
        n_attempts=selection.n_attempts,
        n_generated_via_fewshot=n_generated_via_fewshot,
        fully_satisfied=fully_satisfied,
    )
