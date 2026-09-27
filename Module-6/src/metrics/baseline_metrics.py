"""
Aggregates scripts/05_run_baselines.py's raw per-question JSONL records
into the Module 1 (CoT-SC) checklist table:

    Accuracy, EM, F1, TruthfulQA metric, Majority Vote Share, Vote Entropy,
    # Unique Answers, Wrong Consensus Rate, Invalid Answer Rate,
    Average Tokens, Total Tokens, Average Latency

for each (model, dataset, N) — matching the project's metrics-storage
convention (question -> round/sample -> then aggregated up), so nothing
here needs a re-run to add a different rollup later; it all comes from
the already-saved per-sample records.

Notes on two checklist entries that don't have a separate implementation
--------------------------------------------------------------------------
- "TruthfulQA metric": the DIGRA paper reports plain majority-vote
  accuracy uniformly across all its benchmarks (NQ/BoolQ/TruthfulQA/
  MMLU/GSM8K) — there is no TruthfulQA-specific scoring function in the
  paper to reproduce. `truthfulqa_metric` below is therefore an alias for
  `accuracy`, computed only when dataset == "truthfulqa", so the
  checklist column is never silently blank but also never implies a
  metric that doesn't exist in the source material.
- EM/F1 are computed against the MAJORITY-VOTE answer (the thing that
  would actually be reported as "the model's answer" at this N), not
  averaged per-sample — per-sample EM/F1 would reward high answer
  diversity even when the consensus answer is wrong, which is the
  opposite of what CoT-SC is trying to measure.
"""

from __future__ import annotations

from typing import Optional

from src.metrics.text_metrics import exact_match, token_f1


def aggregate_baseline_records(records: list, k: int) -> dict:
    """
    `records`: list of dicts as written by
    src/baselines/orchestration.py's `_result_to_record` (one per
    question), all for the same (dataset, model). `k`: which
    self-consistency sample count to aggregate (must be a key present in
    every record's "votes" dict, i.e. k <= n_samples_requested and k was
    in `sc_ks` when scripts/05_run_baselines.py was run).

    Returns None if no record has a vote at this k (never silently
    reports an empty-average as 0.0).
    """
    k_key = str(k)
    usable = [r for r in records if k_key in r.get("votes", {})]
    if not usable:
        return None

    dataset = usable[0]["dataset"]
    model = usable[0]["model"]

    n_q = len(usable)
    accuracy = sum(r["votes"][k_key]["majority_is_correct"] for r in usable) / n_q
    majority_vote_share = sum(r["votes"][k_key]["majority_vote_share"] for r in usable) / n_q
    vote_entropy = sum(r["votes"][k_key]["vote_entropy"] for r in usable) / n_q
    n_unique_answers = sum(r["votes"][k_key]["n_unique_answers"] for r in usable) / n_q

    genuine_majority = [r for r in usable if r["votes"][k_key]["majority_vote_share"] > 0.5]
    wrong_consensus_rate = (
        sum(1 for r in genuine_majority if not r["votes"][k_key]["majority_is_correct"])
        / len(genuine_majority)
        if genuine_majority else None
    )

    total_invalid_samples = sum(
        sum(1 for s in r["samples"][:k] if s["is_invalid"]) for r in usable
    )
    invalid_answer_rate = total_invalid_samples / (n_q * k)

    em_values, f1_values = [], []
    for r in usable:
        majority_answer = r["votes"][k_key]["majority_answer"]
        gold = r["gold_answer"]
        alts = r.get("gold_answer_alternatives", [])
        em_values.append(exact_match(majority_answer, gold, alts))
        f1_values.append(token_f1(majority_answer, gold, alts))
    em = sum(em_values) / len(em_values)
    f1 = sum(f1_values) / len(f1_values)

    token_counts = [
        s["n_tokens"] for r in usable for s in r["samples"][:k] if s["n_tokens"] is not None
    ]
    average_tokens: Optional[float] = sum(token_counts) / len(token_counts) if token_counts else None
    total_tokens: Optional[int] = sum(token_counts) if token_counts else None

    latencies = [r["latency_seconds"] for r in usable]
    average_latency = sum(latencies) / len(latencies)

    result = {
        "dataset": dataset,
        "model": model,
        "n_samples": k,
        "n_questions": n_q,
        "accuracy": accuracy,
        "em": em,
        "f1": f1,
        "truthfulqa_metric": accuracy if dataset == "truthfulqa" else None,
        "majority_vote_share": majority_vote_share,
        "vote_entropy": vote_entropy,
        "n_unique_answers": n_unique_answers,
        "wrong_consensus_rate": wrong_consensus_rate,
        "invalid_answer_rate": invalid_answer_rate,
        "average_tokens": average_tokens,
        "total_tokens": total_tokens,
        "average_latency_seconds": average_latency,
    }
    if average_tokens is None:
        result["_note"] = (
            "average_tokens/total_tokens are null: no sample had token_logprobs "
            "populated. Pass generation.logprobs_topk through to run_cot_sc's "
            "logprobs_topk to enable real token counts."
        )
    return result
