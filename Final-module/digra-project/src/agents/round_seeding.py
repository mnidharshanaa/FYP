"""
Round-1 seeding from response pools.

Implements the round-1 initialization described around Table I / Appendix
B-1 of the DIGRA paper: for a hallucination setup like (2 incorrect, 1
correct), 2 agents are seeded with a pre-collected incorrect response and
1 with a pre-collected correct response — no model call happens for round
1 in these setups, only for "standard debate" (see
src/agents/debate_generation.py for that path).

Deliberately has zero LLMClient dependency — this is pure selection logic
over already-built pools, so it's tested exhaustively here with no GPU
involved.
"""

from __future__ import annotations

import random


class InsufficientPoolError(ValueError):
    """Raised when a pool is completely empty but responses are needed from it."""


def seed_round_one_from_pool(
    n_incorrect: int,
    n_correct: int,
    correct_pool: list,
    incorrect_pool: list,
    rng: random.Random,
) -> list:
    """
    Return a list of `n_incorrect + n_correct` response texts, shuffled so
    that which agent index ends up correct/incorrect isn't positionally
    predictable (avoids accidentally confounding "agent 0" with "always
    seeded correct" across an entire experiment).

    FARM provides exactly 3 logical-appeal passages per question (see
    src/data/farm_schema.py), and the correct pool is built to size
    `n_all` (5 by default, see src/data/pool_generation.py). If a setup
    needs more agents seeded from a pool than that pool has distinct
    entries (e.g. a 5-incorrect-agent setup drawing from only 3 appeals),
    we sample WITH replacement rather than fail — documented here as a
    deliberate, visible deviation rather than a silent one.
    """
    if n_incorrect > 0 and not incorrect_pool:
        raise InsufficientPoolError(
            f"Need {n_incorrect} incorrect seed(s) but incorrect_pool is empty."
        )
    if n_correct > 0 and not correct_pool:
        raise InsufficientPoolError(
            f"Need {n_correct} correct seed(s) but correct_pool is empty."
        )

    incorrect_choices = _sample_n(incorrect_pool, n_incorrect, rng)
    correct_choices = _sample_n(correct_pool, n_correct, rng)

    combined = incorrect_choices + correct_choices
    rng.shuffle(combined)
    return combined


def _sample_n(pool: list, n: int, rng: random.Random) -> list:
    if n == 0:
        return []
    if len(pool) >= n:
        return rng.sample(pool, n)
    # Pool smaller than needed — sample with replacement, deterministically
    # given the seeded rng.
    return [rng.choice(pool) for _ in range(n)]
