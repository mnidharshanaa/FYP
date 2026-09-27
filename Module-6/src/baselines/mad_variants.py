"""
Sparse and Random MAD baselines (configs/base.yaml's "mad_sparse_half" /
"mad_random"), built as alternate `communication_fn`s plugged into
src/agents/debate.py's `run_debate` — the round loop, prompt construction,
and grading are ALL identical to Standard MAD (Module 2); only "who does
agent i see this round" differs. This is exactly the reuse
src/agents/debate.py's `CommunicationFn` hook was built for (see its
module docstring: "DIGRA's dynamic topology ... reuses run_debate's
round-2+ loop structure but restricts other_responses per agent to a
selected subset" — sparse/random are two more instances of that same
pattern, simpler than DIGRA's IGR-driven one).

Both variants only make sense for setup="standard" (Table II reports
these baselines under genuine, unseeded round-1 generation — the
hallucination_setups pool-seeding in configs/base.yaml is the
Woozle-effect propagation-analysis concern, Modules 2/3, not a Table II
baseline comparison concern).
"""

from __future__ import annotations

import random
from typing import Optional

from src.agents.debate import CommunicationFn, DebateResult, run_debate
from src.llm.client import LLMClient

VALID_VARIANTS = ("mad_sparse_half", "mad_random")


def build_sparse_communication_fn(n_agents: int, degree: float, seed: int) -> CommunicationFn:
    """
    Fixed sparse topology (Li et al., 2024 — DIGRA paper's ref [39]): each
    agent is assigned d = round(degree * (n_agents - 1)) partners, chosen
    ONCE per debate (not re-drawn every round) uniformly at random from
    the other n_agents - 1 agents. `degree` is D = d / (Na - 1) in the
    paper's own notation (configs/base.yaml's mad_variants.sparse_degree;
    0.5 for "mad_sparse_half").
    """
    if not (0.0 <= degree <= 1.0):
        raise ValueError(f"degree must be in [0, 1], got {degree}")

    rng = random.Random(seed)
    d = round(degree * (n_agents - 1))
    d = max(0, min(d, n_agents - 1))

    partner_indices = {}
    for agent_id in range(n_agents):
        others = [j for j in range(n_agents) if j != agent_id]
        partner_indices[agent_id] = sorted(rng.sample(others, d)) if d > 0 else []

    def _fn(agent_id: int, round_idx: int, prev_round_responses: list) -> list:
        return [prev_round_responses[j] for j in partner_indices[agent_id]]

    return _fn


def build_random_communication_fn(n_agents: int, seed: int) -> CommunicationFn:
    """
    Random MAD (paper's "MAD(random)"): each agent, EVERY round,
    independently draws a fresh random NUMBER (0..n_agents-1) and
    IDENTITY of partners — matching the paper's "introduces randomness by
    randomly choosing both the number and identity of communication
    partners". Re-drawn per (agent_id, round_idx), unlike the sparse
    variant's fixed-once assignment, so successive rounds don't repeat
    the same random partners. Still fully reproducible from `seed` alone:
    each (agent_id, round_idx) pair maps to its own combined integer seed
    (seed + agent_id*100_003 + round_idx*1_009), so two agents/rounds
    never accidentally draw identical random state.
    """
    def _fn(agent_id: int, round_idx: int, prev_round_responses: list) -> list:
        # Combine into a single deterministic int seed (random.Random only
        # accepts int/float/str/bytes, not tuples) — matches
        # src/agents/debate.py's own "seed + round_idx * 1000" style
        # combination for round-level reseeding.
        combined_seed = seed + agent_id * 100_003 + round_idx * 1_009
        rng = random.Random(combined_seed)
        others = [j for j in range(n_agents) if j != agent_id]
        k = rng.randint(0, len(others))
        chosen = sorted(rng.sample(others, k))
        return [prev_round_responses[j] for j in chosen]

    return _fn


def build_communication_fn(
    variant: str, n_agents: int, seed: int, sparse_degree: Optional[float] = None,
) -> CommunicationFn:
    """Dispatch by variant name — the only place that needs to know both exist."""
    if variant == "mad_sparse_half":
        if sparse_degree is None:
            raise ValueError(
                "mad_sparse_half requires sparse_degree "
                "(configs/base.yaml's mad_variants.sparse_degree)"
            )
        return build_sparse_communication_fn(n_agents, degree=sparse_degree, seed=seed)
    if variant == "mad_random":
        return build_random_communication_fn(n_agents, seed=seed)
    raise ValueError(f"Unknown MAD variant {variant!r}. Expected one of {VALID_VARIANTS}")


def run_mad_variant_debate(
    llm: LLMClient,
    variant: str,
    question_id: str,
    question: str,
    gold_answer: str,
    n_agents: int,
    n_rounds: int,
    gold_answer_alternatives: Optional[list] = None,
    seed: int = 0,
    sparse_degree: Optional[float] = None,
    max_tokens: int = 300,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 50,
) -> DebateResult:
    """
    One full debate for one question under a sparse/random MAD topology.
    Delegates entirely to src/agents/debate.py's run_debate (setup=
    "standard" — see module docstring for why) — returns the exact same
    DebateResult shape Standard MAD produces, so every existing consumer
    (src/metrics/standard_mad_adapter.py, propagation_metrics.py,
    scripts/04_generate_report.py) works on these results unmodified.
    """
    if variant not in VALID_VARIANTS:
        raise ValueError(f"Unknown MAD variant {variant!r}. Expected one of {VALID_VARIANTS}")

    communication_fn = build_communication_fn(
        variant, n_agents=n_agents, seed=seed, sparse_degree=sparse_degree,
    )
    return run_debate(
        llm=llm,
        question_id=question_id,
        question=question,
        gold_answer=gold_answer,
        n_agents=n_agents,
        n_rounds=n_rounds,
        setup="standard",
        gold_answer_alternatives=gold_answer_alternatives,
        communication_fn=communication_fn,
        seed=seed,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
