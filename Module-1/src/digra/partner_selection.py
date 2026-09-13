"""
Partner selection (Eq. 9): argmax over candidate partner subsets J.

Caching design (the "store already calculated" requirement)
---------------------------------------------------------------
An agent's own unconditioned entropy H(R_i,t) is fixed for the whole
round — it does not depend on which candidate subset J is being
evaluated. Computing it once per agent per round and passing it in
(rather than recomputing it inside every subset evaluation) is the main
cost saving here, and it's free: H(R_i,t) comes directly from the
logprobs already returned by that agent's own generation call — no
forced-decode needed for it at all (only the *conditional* entropy
H(R_i,t | J), one per candidate subset, needs forced-decoding).

This module takes h_self and per-candidate H(R_j,t) values as already-
computed inputs, and an injected `ig_fn(subset) -> float` callable for
the one thing that genuinely can't be cached across subsets: the
conditional entropy specific to each candidate subset. This keeps the
module: (a) free of any LLMClient dependency, so it's fully unit-testable
with a fake ig_fn, and (b) explicit about exactly where the real cost
lives.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable, Optional

from src.digra.information_gain import compute_igr


@dataclass
class PartnerSelectionResult:
    best_subset: frozenset
    best_igr: float
    all_scores: dict   # subset (frozenset) -> igr score, kept for inspection/debugging


def enumerate_candidate_subsets(candidate_ids: list, max_subset_size: Optional[int] = None):
    """
    All non-empty subsets of candidate_ids, optionally capped at
    max_subset_size (used for e.g. Fig. 8(c)-style experiments capping
    agents to 2 communication partners as agent count scales — see
    configs/base.yaml's digra.max_partner_set_size).
    """
    n = len(candidate_ids)
    max_size = max_subset_size if max_subset_size is not None else n
    max_size = min(max_size, n)

    for size in range(1, max_size + 1):
        for combo in itertools.combinations(candidate_ids, size):
            yield frozenset(combo)


def select_best_partners(
    candidate_ids: list,
    entropy_by_agent: dict,          # agent_id -> H(R_agent,t), already computed
    ig_fn: Callable[[frozenset], float],   # subset -> IG for the requesting agent
    alpha: float,
    max_subset_size: Optional[int] = None,
) -> PartnerSelectionResult:
    """
    Brute-force search over candidate subsets (see module docstring for
    why brute force is fine at the agent counts this project runs at —
    3-5 agents means at most 2^4-1=15 subsets, trivial to enumerate; this
    does NOT scale gracefully to large agent counts, a known, documented
    limitation rather than a hidden one).
    """
    if not candidate_ids:
        raise ValueError("select_best_partners requires at least one candidate")

    scores = {}
    best_subset, best_igr = None, float("-inf")

    for subset in enumerate_candidate_subsets(candidate_ids, max_subset_size):
        ig = ig_fn(subset)
        mean_h_j = sum(entropy_by_agent[j] for j in subset) / len(subset)
        igr = compute_igr(ig, mean_h_j, alpha)
        scores[subset] = igr
        if igr > best_igr:
            best_igr, best_subset = igr, subset

    return PartnerSelectionResult(best_subset=best_subset, best_igr=best_igr, all_scores=scores)
