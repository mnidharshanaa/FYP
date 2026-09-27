"""
Pure pool-selection bookkeeping (Appendix B-3, steps 1-2).

Deliberately has zero dependency on any LLM client — this is the part of
pool-building that's just arithmetic and selection over already-generated
text, so it's tested exhaustively here with no GPU involved. The orchestration
that actually calls a model to produce the 50 samples and any additional
few-shot generations lives in src/data/pool_generation.py.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from src.metrics.grading import extract_final_answer, is_correct


@dataclass
class PoolSelectionResult:
    correct_texts: list                # texts retained from round-1 sampling (<= n_all)
    n_correct_sampled: int             # n1 in the paper's notation
    n_attempts: int                    # total samples drawn (50 in the paper)
    difficulty: float                  # n1 / n_attempts — used later for Fig. 4-style analysis
    n_needed: int                      # additional correct responses still required to reach n_all


def difficulty_score(n_correct: int, n_attempts: int) -> float:
    if n_attempts == 0:
        raise ValueError("n_attempts must be > 0")
    return n_correct / n_attempts


def select_correct_pool(
    sampled_texts: list,
    gold_answer: str,
    alternatives: list,
    n_all: int,
    rng: random.Random,
) -> PoolSelectionResult:
    """
    Grade each of the already-generated `sampled_texts`, then apply Appendix
    B-3 step 2: if at least n_all are correct, randomly retain exactly n_all
    of them; otherwise keep every correct one found and report how many
    more are still needed (step 3 territory, handled by the caller).
    """
    graded_correct = [
        text for text in sampled_texts
        if is_correct(extract_final_answer(text), gold_answer, alternatives)
    ]
    n1 = len(graded_correct)
    n_attempts = len(sampled_texts)
    diff = difficulty_score(n1, n_attempts)

    if n1 >= n_all:
        kept = rng.sample(graded_correct, n_all)
        return PoolSelectionResult(
            correct_texts=kept, n_correct_sampled=n1,
            n_attempts=n_attempts, difficulty=diff, n_needed=0,
        )

    return PoolSelectionResult(
        correct_texts=graded_correct, n_correct_sampled=n1,
        n_attempts=n_attempts, difficulty=diff, n_needed=n_all - n1,
    )
