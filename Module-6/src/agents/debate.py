"""
Core debate engine (Standard MAD — fully connected communication topology).

This is the paper's baseline debate mechanic: N agents, n_rounds rounds,
every agent sees every other agent's previous-round response each round.
DIGRA's dynamic topology (Module 5) will reuse `run_debate`'s round-2+
loop structure but restrict `other_responses` per agent to a selected
subset instead of "all other agents" — see the `communication_fn` hook
below, added specifically so Module 5 doesn't have to duplicate this loop.

Depends only on the LLMClient interface, so every branch here is tested
in tests/test_debate.py using FakeLLMClient — no GPU required.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Optional

from src.agents.prompts import build_debate_round_prompt, build_neutral_prompt
from src.agents.round_seeding import seed_round_one_from_pool
from src.llm.client import LLMClient
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Signature for a pluggable "who does agent i talk to this round" function:
# (agent_id, round_idx, prev_round_responses) -> list of response texts
# `prev_round_responses` is agent-indexed. Default (fully connected) is
# "everyone except agent_id" — see `_fully_connected_partners` below.
CommunicationFn = Callable[[int, int, list], list]


@dataclass
class DebateResult:
    question_id: str
    setup: str                    # e.g. "2,1" or "standard" — string form for easy CSV/JSON storage
    n_agents: int
    n_rounds: int
    agent_responses: list         # shape (n_agents, n_rounds), agent_responses[i][t] = text
    gold_answer: str
    gold_answer_alternatives: list = field(default_factory=list)


def _fully_connected_partners(agent_id: int, round_idx: int, prev_round_responses: list) -> list:
    return [resp for j, resp in enumerate(prev_round_responses) if j != agent_id]


def generate_round_one_standard(
    llm: LLMClient,
    question: str,
    n_agents: int,
    seed: int = 0,
    max_tokens: int = 300,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 50,
) -> list:
    """
    'Standard Debate' round-1 initialization (Table I's bottom row): no
    seeding from pools — every agent independently generates a genuine
    fresh response to the neutral prompt.

    `max_tokens`/etc should be sourced from configs/base.yaml's
    `generation` section by the caller, not left at these defaults in a
    real run — see src/data/pool_generation.py's docstring for the
    context-length failure this matters for.
    """
    prompt = build_neutral_prompt(question)
    results = llm.generate(
        prompt, n=n_agents, seed=seed,
        max_tokens=max_tokens, temperature=temperature, top_p=top_p, top_k=top_k,
    )
    if len(results) != n_agents:
        raise RuntimeError(
            f"Expected {n_agents} round-1 responses from LLMClient, got {len(results)}."
        )
    return [r.text for r in results]


def run_debate(
    llm: LLMClient,
    question_id: str,
    question: str,
    gold_answer: str,
    n_agents: int,
    n_rounds: int,
    setup,                                   # tuple[int, int] or the literal string "standard"
    gold_answer_alternatives: Optional[list] = None,
    correct_pool: Optional[list] = None,     # required unless setup == "standard"
    incorrect_pool: Optional[list] = None,   # required unless setup == "standard"
    communication_fn: CommunicationFn = _fully_connected_partners,
    seed: int = 0,
    max_tokens: int = 300,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 50,
) -> DebateResult:
    """
    Run one full debate for one question.

    `setup` is either the literal string "standard" (genuine fresh round-1
    generation) or a (n_incorrect, n_correct) tuple that must sum to
    n_agents (pool-seeded round 1, per Appendix B-1).

    `communication_fn` defaults to fully-connected (Standard MAD). Module 5
    (DIGRA) passes a different function here that restricts each agent's
    visible partners to its IGR-selected subset — the rest of this loop is
    identical between Standard MAD and DIGRA, which is why it's factored
    out as a hook rather than duplicated.

    `max_tokens`/`temperature`/`top_p`/`top_k` should be sourced from
    configs/base.yaml's `generation` section by the caller.

    Note on seeding: round 2+ generation is seeded once per round
    (`seed + round_idx * 1000`), not per agent as an earlier unbatched
    implementation did — batching all agents' prompts into one
    generate_batch() call means they share one SamplingParams seed. This
    is a deliberate, documented trade-off for the batching performance fix
    (see generate_batch's docstring), not expected to bias results at the
    scale this project runs at (4 seeds already provide seed-level
    variance across whole debates).
    """
    gold_answer_alternatives = gold_answer_alternatives or []

    if setup == "standard":
        round1 = generate_round_one_standard(
            llm, question, n_agents, seed=seed,
            max_tokens=max_tokens, temperature=temperature, top_p=top_p, top_k=top_k,
        )
        setup_label = "standard"
    else:
        n_incorrect, n_correct = setup
        if n_incorrect + n_correct != n_agents:
            raise ValueError(
                f"setup {setup} sums to {n_incorrect + n_correct}, "
                f"expected n_agents={n_agents}"
            )
        rng = random.Random(seed)
        round1 = seed_round_one_from_pool(
            n_incorrect, n_correct, correct_pool or [], incorrect_pool or [], rng,
        )
        setup_label = f"{n_incorrect},{n_correct}"

    # agent_responses[i] accumulates round texts for agent i, in round order.
    agent_responses = [[text] for text in round1]

    for round_idx in range(1, n_rounds):
        prev_round_responses = [agent_responses[i][-1] for i in range(n_agents)]

        prompts = []
        for agent_id in range(n_agents):
            partner_responses = communication_fn(agent_id, round_idx, prev_round_responses)
            prompts.append(
                build_debate_round_prompt(
                    question=question,
                    own_previous_response=prev_round_responses[agent_id],
                    other_responses=partner_responses,
                )
            )

        # Single batched call for the whole round — every agent's prompt is
        # different, so this is generate_batch (many distinct prompts, one
        # completion each), not generate's n>1-of-one-prompt batching. See
        # LLMClient.generate_batch's docstring for why this matters: the
        # original one-call-per-agent loop left the GPU at batch size 1 for
        # the entire debate, which was the actual cause of multi-day runtimes.
        results = llm.generate_batch(
            prompts, seed=seed + round_idx * 1000,
            max_tokens=max_tokens, temperature=temperature, top_p=top_p, top_k=top_k,
        )
        new_round_texts = [r.text for r in results]

        for agent_id in range(n_agents):
            agent_responses[agent_id].append(new_round_texts[agent_id])

    return DebateResult(
        question_id=question_id,
        setup=setup_label,
        n_agents=n_agents,
        n_rounds=n_rounds,
        agent_responses=agent_responses,
        gold_answer=gold_answer,
        gold_answer_alternatives=gold_answer_alternatives,
    )
