"""
Adapter: Standard MAD DebateResult -> the agent_records shape
src/metrics/propagation_metrics.py already expects.

Standard MAD (src/agents/debate.py's DebateResult) and DIGRA
(src/digra/digra_debate.py's DigraDebateResult) save different schemas —
Standard MAD never pre-computes correctness or uses 1-indexed round_idx
records, it's just agent_responses[i][t] = text. Rather than duplicate
the MA/MR/IMR/CR/accuracy formulas for a second schema, this adapter
reshapes Standard MAD data into the exact shape compute_propagation_metrics
and compute_final_accuracy already consume, so both data sources share
one tested implementation of the metrics themselves.
"""

from __future__ import annotations

from src.metrics.grading import extract_final_answer, is_correct


def adapt_standard_mad_debate(debate: dict) -> dict:
    gold = debate["gold_answer"]
    alternatives = debate.get("gold_answer_alternatives", [])

    agent_records = []
    for agent_responses in debate["agent_responses"]:
        history = []
        for t, text in enumerate(agent_responses):
            answer = extract_final_answer(text)
            correct = is_correct(answer, gold, alternatives)
            history.append({
                "round_idx": t + 1,  # Standard MAD is 0-indexed by list position; metrics expect 1-indexed
                "response_text": text,
                "extracted_answer": answer,
                "is_correct": correct,
            })
        agent_records.append(history)

    return {"agent_records": agent_records}


def adapt_standard_mad_debates(debates: list) -> list:
    return [adapt_standard_mad_debate(d) for d in debates]
