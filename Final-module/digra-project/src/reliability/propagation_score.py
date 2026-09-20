"""
Propagation Control Module — the scoring and decision logic only
(Sections 2-10 of the frozen final-module spec). Deliberately has ZERO
dependency on any LLM, dataset, or debate loop: every function here is
pure math over already-computed U/R values, so it's fully testable and
frozen before a single extra model call gets made.

Terminology, matching the spec exactly:
  U   = normalized information utility (from DIGRA's IGR)
  R   = per-agent, per-round semantic reliability (Base 2-style
        entailment clustering — see src/reliability/entailment.py)
  lam = the combination weight (NOT DIGRA's own alpha — kept as a
        distinctly-named parameter everywhere, per the spec's explicit
        instruction to never conflate the two)
  PS  = U^lam * R^(1-lam)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

Decision = Literal["propagate", "verify", "suppress"]
Quadrant = Literal["propagate", "verify", "suppress", "low_priority"]


def normalize_utility(igr: float, igr_min: float, igr_max: float, epsilon: float = 1e-9) -> float:
    """
    U = (IGR - IGR_min) / (IGR_max - IGR_min + epsilon), clipped to [0, 1].

    `igr_min`/`igr_max` should come from the same pool of IGR values this
    `igr` was drawn from (e.g. all candidate partners for one agent this
    round) — this is a min-max normalization over that pool, not a
    universal constant, so the caller controls the comparison set.
    Clipping guards against float roundoff pushing a value fractionally
    outside [0, 1] (e.g. igr == igr_max should give exactly 1.0, not
    1.0000000002), not against genuinely out-of-pool inputs.
    """
    if igr_max < igr_min:
        raise ValueError(f"igr_max ({igr_max}) must be >= igr_min ({igr_min})")
    raw = (igr - igr_min) / (igr_max - igr_min + epsilon)
    return max(0.0, min(1.0, raw))


def propagation_score(u: float, r: float, lam: float) -> float:
    """
    PS = U^lam * R^(1-lam) (Section 4). Geometric, not arithmetic,
    combination — a message scores near-zero if EITHER signal is
    near-zero, which is the whole point of using this form over a
    weighted average (see the spec's worked example: U=0.9,R=0.2 scores
    0.424, not the arithmetic mean's 0.55).

    lam=0 reduces to PS=R (reliability-only policy); lam=1 reduces to
    PS=U (utility-only policy, i.e. plain DIGRA) — both handled correctly
    by Python's `**` (x**0 == 1 for any x, including x=0), so no special
    casing is needed for the boundary lam values.
    """
    if not (0.0 <= u <= 1.0):
        raise ValueError(f"u must be in [0, 1], got {u}")
    if not (0.0 <= r <= 1.0):
        raise ValueError(f"r must be in [0, 1], got {r}")
    if not (0.0 <= lam <= 1.0):
        raise ValueError(f"lam must be in [0, 1], got {lam}")
    return (u ** lam) * (r ** (1.0 - lam))


def classify_decision(ps: float, u: float, tau_p: float, tau_u: float) -> Decision:
    """
    Section 8's main decision policy:
      PROPAGATE : PS >= tau_p
      VERIFY    : PS <  tau_p AND U >= tau_u
      SUPPRESS  : PS <  tau_p AND U <  tau_u
    """
    if ps >= tau_p:
        return "propagate"
    if u >= tau_u:
        return "verify"
    return "suppress"


def classify_quadrant(u: float, r: float, u_threshold: float = 0.5, r_threshold: float = 0.5) -> Quadrant:
    """
    Section 10's four-quadrant analysis — a SEPARATE, simpler
    classification from `classify_decision`: a plain midpoint split on U
    and R independently (no PS, no tau_p/tau_u), used purely to report
    how communications distribute across the four conceptual categories:

                        RELIABILITY
                     LOW           HIGH
                  +----------+-----------+
              HIGH| VERIFY   | PROPAGATE |
       UTILITY    +----------+-----------+
               LOW| SUPPRESS |LOW_PRIORITY|
                  +----------+-----------+

    Do not use this for the actual propagation decision in the pipeline —
    that's `classify_decision`'s job. This is a reporting/diagnostic view
    of the same (U, R) pairs, answering "why do we need both signals?",
    not a second decision policy.
    """
    u_high = u >= u_threshold
    r_high = r >= r_threshold
    if u_high and r_high:
        return "propagate"
    if u_high and not r_high:
        return "verify"
    if not u_high and r_high:
        return "low_priority"
    return "suppress"


@dataclass(frozen=True)
class PropagationDecisionRecord:
    """Bundles one edge's full scoring+decision trail — the minimal unit
    Section 22's aggregate metrics (propagate/verify/suppress counts,
    PS_mean, PS_correct_mean, etc.) are computed over."""

    u: float
    r: float
    lam: float
    ps: float
    tau_p: float
    tau_u: float
    decision: Decision
    quadrant: Quadrant


def score_and_decide(
    u: float, r: float, lam: float, tau_p: float, tau_u: float,
    quadrant_u_threshold: float = 0.5, quadrant_r_threshold: float = 0.5,
) -> PropagationDecisionRecord:
    """Convenience wrapper computing PS, the main decision, and the
    quadrant label in one call, so orchestration code never has to
    remember to keep all three in sync by hand."""
    ps = propagation_score(u, r, lam)
    decision = classify_decision(ps, u, tau_p, tau_u)
    quadrant = classify_quadrant(u, r, quadrant_u_threshold, quadrant_r_threshold)
    return PropagationDecisionRecord(
        u=u, r=r, lam=lam, ps=ps, tau_p=tau_p, tau_u=tau_u,
        decision=decision, quadrant=quadrant,
    )
