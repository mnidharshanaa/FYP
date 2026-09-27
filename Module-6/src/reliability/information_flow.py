"""
Information-flow classification (Sections 18-19): for every communication
edge, classify what happened to correctness as information moved from a
source's state to a target's resulting state, and aggregate the
per-round rates the spec calls the most informative metric in the whole
module — more informative than final accuracy alone, since it shows
*which direction* correctness is moving, not just the net result.
"""

from __future__ import annotations

from typing import Literal, Optional

FlowLabel = Literal["C->C", "C->W", "W->C", "W->W"]


def classify_information_flow(origin_correct: bool, result_correct: bool) -> FlowLabel:
    """
    origin_correct: was the SOURCE's information correct.
    result_correct: was the TARGET correct after incorporating it.

    W->W (origin wrong, still wrong after) is flagged by the spec as the
    single most dangerous category — an error being actively maintained/
    reinforced, not just failing to improve. W->C is the beneficial
    correction case DIGRA/DIGRA+reliability is trying to maximize.
    """
    if origin_correct and result_correct:
        return "C->C"
    if origin_correct and not result_correct:
        return "C->W"
    if not origin_correct and result_correct:
        return "W->C"
    return "W->W"


def compute_flow_rates(flows: list) -> dict:
    """
    `flows`: list of FlowLabel strings (e.g. from classify_information_flow
    over every edge in one round). Returns the three named rates from
    Section 19, each None (never a fabricated 0.0) when its denominator
    is empty — e.g. WWR/WCR are undefined for a round with zero
    wrong-origin communications, exactly like propagation_metrics.py's
    existing MR/IMR/CR None-when-empty convention elsewhere in this
    project, so every consumer of these rates already knows what a None
    means and doesn't need a second convention to learn.

    Returns:
      {"WWR": wrong-to-wrong rate | None,
       "WCR": wrong-to-correct correction rate | None,
       "CWR": correct-to-wrong corruption rate | None,
       "counts": {"C->C": int, "C->W": int, "W->C": int, "W->W": int}}
    """
    counts = {"C->C": 0, "C->W": 0, "W->C": 0, "W->W": 0}
    for f in flows:
        if f not in counts:
            raise ValueError(f"Unknown flow label {f!r}, expected one of {list(counts)}")
        counts[f] += 1

    wrong_origin = counts["W->C"] + counts["W->W"]
    correct_origin = counts["C->C"] + counts["C->W"]

    wwr = counts["W->W"] / wrong_origin if wrong_origin > 0 else None
    wcr = counts["W->C"] / wrong_origin if wrong_origin > 0 else None
    cwr = counts["C->W"] / correct_origin if correct_origin > 0 else None

    return {"WWR": wwr, "WCR": wcr, "CWR": cwr, "counts": counts}


def compute_flow_rates_by_round(flows_by_round: dict) -> dict:
    """`flows_by_round`: {round_idx: [FlowLabel, ...]}. Returns
    {round_idx: compute_flow_rates(...)} — Section 19's per-round table."""
    return {round_idx: compute_flow_rates(flows) for round_idx, flows in flows_by_round.items()}


def compute_hpr(comms: list) -> Optional[float]:
    """
    Hallucination Propagation Rate — a strictly stronger propagation
    signal than plain C->W, added per the project's own extension beyond
    either base paper. Requires THREE time points, not two:
      - origin_correct:         source's correctness at round t
      - target_correct_before:  TARGET's own correctness at round t,
                                 i.e. BEFORE this communication — a
                                 distinct quantity from origin_correct
                                 (same round, different agent) that
                                 classify_information_flow's 2-state
                                 origin/result labels don't capture at all
      - result_correct:         target's correctness at round t+1, AFTER
                                 incorporating the communication

    Among "W->C selected opportunities" (source wrong, target ALREADY
    correct before this communication — a genuine at-risk pairing), HPR
    is the fraction where the target went on to become wrong. This is a
    stronger claim than C->W: it specifically isolates cases where a
    previously-correct agent was put at risk by a wrong source and that
    risk materialized, rather than any correct-to-wrong transition
    regardless of what caused it.

    `comms`: list of dicts (or CommunicationRecord-shaped objects) with
    origin_correct, target_correct_before, result_correct fields. None
    (never a fabricated 0.0/1.0) if there are no such opportunities in
    the given data.
    """
    opportunities = [
        c for c in comms
        if c["origin_correct"] is False and c.get("target_correct_before") is True
        and c["result_correct"] is not None
    ]
    if not opportunities:
        return None
    harm_realized = sum(1 for c in opportunities if c["result_correct"] is False)
    return harm_realized / len(opportunities)


def compute_bhp(comms: list) -> Optional[float]:
    """
    Blocked Hallucination Propagation rate — of the same "W->C selected
    opportunities" HPR is computed over (source wrong, target already
    correct before this communication), what fraction were BLOCKED
    (final decision != "propagate") by the reliability gate, rather than
    let through unconditionally. This directly answers "did the gate
    actually intervene on the risky cases," independent of whether the
    outcome is even known yet — unlike HPR, BHP doesn't require
    result_correct, since it's about the DECISION made, not the observed
    aftermath.

    None (never a fabricated 0.0/1.0) if there are no such opportunities.
    """
    opportunities = [
        c for c in comms
        if c["origin_correct"] is False and c.get("target_correct_before") is True
    ]
    if not opportunities:
        return None
    blocked = sum(1 for c in opportunities if c["decision"] != "propagate")
    return blocked / len(opportunities)
