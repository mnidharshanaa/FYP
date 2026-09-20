"""
Aggregates scripts/10_run_propagation_control.py's raw per-question JSONL
output into the report tables the frozen spec calls for:

  - lambda_ablation_table (Section 24): one row per lambda value —
    accuracy, MR/IMR, information-flow rates, RAG rate, token cost.
  - four_quadrant_table (Section 10): how communications distribute
    across propagate/verify/suppress/low_priority, with accuracy and
    hallucination-flow rates PER quadrant — the experiment that answers
    "why do we need both U and R?".
  - rag_effect_summary (Section 13): RAG trigger rate, verify->propagate
    / verify->suppress rates, and the average R/PS shift RAG produced.

Deliberately NOT built yet (disclosed, not hidden): PS-validity binning
(Section 23) and the calibration/test-split machinery (Section 7) — see
scripts/11_generate_propagation_report.py's module docstring.
"""

from __future__ import annotations

import json
from typing import Optional


def load_propagation_records(paths: list) -> list:
    records = []
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def _all_communication_records(debates: list) -> list:
    return [c for d in debates for c in d["communication_records"]]


def _all_rag_records(debates: list) -> list:
    return [r for d in debates for r in d["rag_verification_records"]]


def lambda_ablation_row(debates: list, lam: float) -> Optional[dict]:
    """
    One row of Section 24's table for a single lambda value. `debates`
    must all share this lambda (the caller groups by lambda before
    calling this — see lambda_ablation_table). Returns None if `debates`
    is empty rather than fabricating a row of zeros.
    """
    if not debates:
        return None

    from src.metrics.propagation_metrics import compute_final_accuracy, compute_propagation_metrics
    from src.reliability.information_flow import compute_flow_rates

    accuracy = compute_final_accuracy(debates)
    prop_metrics = compute_propagation_metrics(debates)

    comms = _all_communication_records(debates)
    flows = [c["information_flow"] for c in comms if c["information_flow"] is not None]
    flow_rates = compute_flow_rates(flows)

    verify_triggered = sum(1 for c in comms if c["pre_rag_decision"] == "verify")
    rag_rate = verify_triggered / len(comms) if comms else None

    token_counts = [d["total_tokens_partial"] for d in debates]
    avg_tokens = sum(token_counts) / len(token_counts) if token_counts else None

    return {
        "lambda": lam,
        "n_debates": len(debates),
        "accuracy": accuracy,
        "MA1": prop_metrics["MA"].get(1),
        "MA_final": prop_metrics["MA"].get(max(prop_metrics["MA"])) if prop_metrics["MA"] else None,
        "MR_final": prop_metrics["MR"].get(max(prop_metrics["MR"])) if prop_metrics.get("MR") else None,
        "IMR_final": prop_metrics["IMR"].get(max(prop_metrics["IMR"])) if prop_metrics.get("IMR") else None,
        "WWR": flow_rates["WWR"],
        "WCR": flow_rates["WCR"],
        "CWR": flow_rates["CWR"],
        "n_candidate_edges": len(comms),
        "rag_trigger_rate": rag_rate,
        "avg_tokens_partial": avg_tokens,
    }


def lambda_ablation_table(debates: list) -> list:
    """Groups `debates` by their communication_records' lambda value
    (every edge in a debate shares one lambda, set for the whole run —
    read from the data itself rather than trusted filenames) and returns
    one row per distinct lambda, sorted ascending."""
    by_lambda: dict = {}
    for d in debates:
        comms = d.get("communication_records", [])
        lam = comms[0]["lam"] if comms else None
        by_lambda.setdefault(lam, []).append(d)

    rows = []
    for lam in sorted(k for k in by_lambda if k is not None):
        row = lambda_ablation_row(by_lambda[lam], lam)
        if row is not None:
            rows.append(row)
    return rows


def four_quadrant_table(debates: list) -> list:
    """
    Section 10's flagship experiment: for every candidate communication
    edge, its (U, R) quadrant vs. whether it ended up correct (via
    result_correct — the RECEIVER's correctness after incorporating it)
    and what fraction were harmful (C->W or W->W, the two flow labels the
    spec calls dangerous). One row per quadrant that has at least one
    edge; a quadrant with zero edges is omitted rather than shown as a
    fabricated all-None row.
    """
    from src.reliability.information_flow import compute_flow_rates

    comms = _all_communication_records(debates)
    by_quadrant: dict = {}
    for c in comms:
        by_quadrant.setdefault(c["quadrant"], []).append(c)

    rows = []
    for quadrant, edges in by_quadrant.items():
        n = len(edges)
        known_correct = [e["result_correct"] for e in edges if e["result_correct"] is not None]
        accuracy_after = sum(known_correct) / len(known_correct) if known_correct else None

        flows = [e["information_flow"] for e in edges if e["information_flow"] is not None]
        flow_rates = compute_flow_rates(flows)
        harmful = sum(1 for f in flows if f in ("C->W", "W->W"))
        harmful_rate = harmful / len(flows) if flows else None

        rag_count = sum(1 for e in edges if e["pre_rag_decision"] == "verify")

        rows.append({
            "quadrant": quadrant,
            "n_edges": n,
            "pct_of_total": n / len(comms) if comms else None,
            "accuracy_after_propagation": accuracy_after,
            "harmful_propagation_rate": harmful_rate,
            "WWR": flow_rates["WWR"],
            "CWR": flow_rates["CWR"],
            "rag_invocation_count": rag_count,
        })

    order = {"propagate": 0, "verify": 1, "low_priority": 2, "suppress": 3}
    rows.sort(key=lambda r: order.get(r["quadrant"], 99))
    return rows


def rag_effect_summary(debates: list) -> Optional[dict]:
    """
    Section 13's RAG success metrics: trigger rate, verify->propagate /
    verify->suppress rates, and the average R/PS shift. Returns None if
    no RAG events exist in `debates` at all (never a fabricated
    zero-effect row for a run where RAG never fired).
    """
    comms = _all_communication_records(debates)
    rag_records = _all_rag_records(debates)
    if not rag_records:
        return None

    n_verify_edges = sum(1 for c in comms if c["pre_rag_decision"] == "verify")
    n_completed = sum(1 for r in rag_records if r["evidence_status"] == "ok")
    n_no_evidence = sum(1 for r in rag_records if r["evidence_status"] == "no_evidence_available")
    n_parse_failed = sum(1 for r in rag_records if r["evidence_status"] == "verification_parse_failed")

    to_propagate = sum(1 for r in rag_records if r["decision_after"] == "propagate")
    to_suppress = sum(1 for r in rag_records if r["decision_after"] == "suppress")

    completed = [r for r in rag_records if r["evidence_status"] == "ok"]
    delta_r = [r["r_after"] - r["r_before"] for r in completed]
    delta_ps = [r["ps_after"] - r["ps_before"] for r in completed]

    return {
        "n_verify_triggering_edges": n_verify_edges,
        "n_rag_events_deduplicated": len(rag_records),
        "n_completed": n_completed,
        "n_no_evidence_available": n_no_evidence,
        "n_verification_parse_failed": n_parse_failed,
        "verify_to_propagate_rate": to_propagate / len(rag_records) if rag_records else None,
        "verify_to_suppress_rate": to_suppress / len(rag_records) if rag_records else None,
        "mean_delta_r": sum(delta_r) / len(delta_r) if delta_r else None,
        "mean_delta_ps": sum(delta_ps) / len(delta_ps) if delta_ps else None,
    }
