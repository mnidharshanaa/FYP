"""
Propagation metrics, computed entirely from already-saved DigraDebateResult
data (src/digra/digra_debate.py's output) — no re-running of any debate.

Implements the base paper's Eq. 1-5 (Mean Accuracy, Misleading Rate, Initial
Misleading Rate, Correction Rate), generalized to work on any variant's
saved agent_records, plus this project's own additions: final accuracy,
RAG contradiction-detector precision/recall (including a "confident wrong"
recall breakdown — the specific case the base paper's entropy-only DIGRA
misses), and cost/overhead stats.
"""

from __future__ import annotations

import math

from src.metrics.grading import compute_em, compute_f1, normalize_answer


def _max_round(results: list) -> int:
    rounds = [
        rec["round_idx"]
        for debate in results
        for agent_hist in debate["agent_records"]
        for rec in agent_hist
    ]
    if not rounds:
        raise ValueError("results contains no agent records")
    return max(rounds)


def compute_propagation_metrics(results: list) -> dict:
    """
    Eq. 1-5: Mean Accuracy (MA), Misleading Rate (MR), Initial Misleading
    Rate (IMR), Correction Rate (CR), computed per round across all debates
    in `results`. Returns {"MA": {round: value}, "MR": {...}, "IMR": {...},
    "CR": {...}, "max_round": int}. A None value means no debates had data
    for that round-transition (e.g. everything early-stopped before then).
    """
    max_round = _max_round(results)
    ma, mr, imr, cr = {}, {}, {}, {}

    for r in range(1, max_round + 1):
        correct_total, total = 0, 0
        mr_num, mr_den = 0, 0
        imr_num, imr_den = 0, 0
        cr_num, cr_den = 0, 0

        for debate in results:
            for agent_hist in debate["agent_records"]:
                by_round = {rec["round_idx"]: rec["is_correct"] for rec in agent_hist}

                if r in by_round:
                    total += 1
                    if by_round[r]:
                        correct_total += 1

                if r >= 2 and (r - 1) in by_round and r in by_round:
                    if by_round[r - 1]:
                        mr_den += 1
                        if not by_round[r]:
                            mr_num += 1
                    else:
                        cr_den += 1
                        if by_round[r]:
                            cr_num += 1

                if r >= 2 and 1 in by_round and r in by_round and by_round[1]:
                    imr_den += 1
                    if not by_round[r]:
                        imr_num += 1

        ma[r] = correct_total / total if total else None
        mr[r] = mr_num / mr_den if mr_den else None
        imr[r] = imr_num / imr_den if imr_den else None
        cr[r] = cr_num / cr_den if cr_den else None

    return {"MA": ma, "MR": mr, "IMR": imr, "CR": cr, "max_round": max_round}


def compute_final_accuracy(results: list) -> float:
    """
    Majority-vote accuracy using each debate's LAST round of responses
    (the round each agent actually stopped at, accounting for individual
    early-freeze via the carried-forward records).
    """
    if not results:
        return None

    n_correct = 0
    for debate in results:
        last_round_entries = []
        for agent_hist in debate["agent_records"]:
            last = max(agent_hist, key=lambda r: r["round_idx"])
            last_round_entries.append((last["extracted_answer"], last["is_correct"]))

        groups: dict = {}
        for answer, correct in last_round_entries:
            key = normalize_answer(answer)
            groups.setdefault(key, []).append(correct)

        majority_key = max(groups, key=lambda k: len(groups[k]))
        majority_correct = sum(groups[majority_key]) / len(groups[majority_key]) >= 0.5
        if majority_correct:
            n_correct += 1

    return n_correct / len(results)


def compute_rag_precision_recall(results: list, entropy_percentile: float = 0.5) -> dict:
    """
    Precision/recall of the RAG contradiction flag against ground-truth
    correctness (is_correct is available for free — no manual labeling
    needed). Also reports "confident_wrong_recall": recall restricted to
    responses that are BOTH incorrect AND low-entropy (below the given
    percentile of all observed entropies) — the specific "faithfulness
    hallucination" case pure entropy-based DIGRA misses, and the number
    that most directly supports the RAG correction's motivation.

    Returns None fields if no variant in `results` used RAG (rag_flagged
    is None throughout) — never silently reports a meaningless 0/0 ratio.
    """
    flagged_records = []
    for debate in results:
        for agent_hist in debate["agent_records"]:
            for rec in agent_hist:
                if rec.get("rag_flagged") is not None:
                    flagged_records.append(rec)

    if not flagged_records:
        return {
            "precision": None, "recall": None, "confident_wrong_recall": None,
            "n_flagged": 0, "n_actual_incorrect": 0, "n_evaluated": 0,
        }

    entropies = sorted(rec["entropy"] for rec in flagged_records)
    threshold_idx = int(len(entropies) * entropy_percentile)
    threshold_idx = min(threshold_idx, len(entropies) - 1)
    entropy_threshold = entropies[threshold_idx]

    tp = fp = fn = 0
    low_h_incorrect_total = 0
    low_h_incorrect_caught = 0

    for rec in flagged_records:
        flagged = rec["rag_flagged"]
        correct = rec["is_correct"]
        if flagged and not correct:
            tp += 1
        elif flagged and correct:
            fp += 1
        elif not flagged and not correct:
            fn += 1

        if not correct and rec["entropy"] <= entropy_threshold:
            low_h_incorrect_total += 1
            if flagged:
                low_h_incorrect_caught += 1

    return {
        "precision": tp / (tp + fp) if (tp + fp) else None,
        "recall": tp / (tp + fn) if (tp + fn) else None,
        "confident_wrong_recall": (
            low_h_incorrect_caught / low_h_incorrect_total if low_h_incorrect_total else None
        ),
        "n_flagged": tp + fp,
        "n_actual_incorrect": tp + fn,
        "n_evaluated": len(flagged_records),
    }


def compute_em_f1(results: list) -> dict:
    """
    Checklist's EM / F1 columns: computed on every agent's FINAL-round
    answer (the round it actually stopped at — same "last round" notion
    compute_final_accuracy uses) against the debate's gold_answer /
    gold_answer_alternatives, averaged across every agent in every
    debate. Most meaningful for extractive QA (NQ); still well-defined
    (just less discriminating) for BoolQ/TruthfulQA, so computed
    unconditionally here — dataset-appropriate framing is a presentation
    choice made by the caller (e.g. only display F1 for the nq_*.jsonl
    file), not something this function needs to know about.
    """
    if not results:
        return {"EM": None, "F1": None, "n": 0}

    em_total, f1_total, n = 0, 0.0, 0
    for debate in results:
        gold = debate["gold_answer"]
        alts = debate.get("gold_answer_alternatives", [])
        for agent_hist in debate["agent_records"]:
            last = max(agent_hist, key=lambda r: r["round_idx"])
            pred = last["extracted_answer"]
            em_total += int(compute_em(pred, gold, alts))
            f1_total += compute_f1(pred, gold, alts)
            n += 1

    return {"EM": em_total / n if n else None, "F1": f1_total / n if n else None, "n": n}


def compute_majority_vote_stats(results: list) -> dict:
    """
    Checklist's "Majority Vote Share", "Vote Entropy", "# Unique
    Answers": computed per-debate at the FINAL round (one vote per
    agent), then averaged across debates.

    Majority vote share = size of the largest answer-group / n_agents
    (1.0 = full consensus). Vote entropy = Shannon entropy (base 2, bits)
    of the normalized answer-group-size distribution (0 = full
    consensus, higher = more disagreement, max = log2(n_agents) when
    every agent disagrees).
    """
    if not results:
        return {"majority_vote_share": None, "vote_entropy": None, "mean_unique_answers": None, "n_debates": 0}

    shares, entropies, unique_counts = [], [], []
    for debate in results:
        answers = []
        for agent_hist in debate["agent_records"]:
            last = max(agent_hist, key=lambda r: r["round_idx"])
            answers.append(normalize_answer(last["extracted_answer"]))

        n_agents = len(answers)
        if n_agents == 0:
            continue
        groups: dict = {}
        for a in answers:
            groups[a] = groups.get(a, 0) + 1

        largest = max(groups.values())
        shares.append(largest / n_agents)
        unique_counts.append(len(groups))

        entropy = 0.0
        for count in groups.values():
            p = count / n_agents
            entropy -= p * math.log2(p)
        entropies.append(entropy)

    return {
        "majority_vote_share": sum(shares) / len(shares) if shares else None,
        "vote_entropy": sum(entropies) / len(entropies) if entropies else None,
        "mean_unique_answers": sum(unique_counts) / len(unique_counts) if unique_counts else None,
        "n_debates": len(shares),
    }


def compute_ig_per_round(results: list) -> dict:
    """
    Checklist requirement: "For IG, keep per edge and per round, not
    just the average." Scans every AgentRoundRecord.candidate_scores
    (every candidate subset evaluated at that selection event — winners
    AND losers, not just the one that got selected) and groups by
    round_idx. Returns {round: {"mean": ..., "min": ..., "max": ...,
    "n_edges": ...}}. Score semantics depend on the variant that
    produced `results` (raw IG for "dig", IGR for "digra*") — this
    function is variant-agnostic, same as compute_communication_stats.
    """
    by_round: dict = {}
    for debate in results:
        for agent_hist in debate["agent_records"]:
            for rec in agent_hist:
                scores = rec.get("candidate_scores")
                if not scores:
                    continue
                by_round.setdefault(rec["round_idx"], []).extend(scores.values())

    return {
        r: {
            "mean": sum(vals) / len(vals),
            "min": min(vals),
            "max": max(vals),
            "n_edges": len(vals),
        }
        for r, vals in sorted(by_round.items())
    }


def compute_token_stats(results: list) -> dict:
    """Checklist's "Average Tokens" / "Total Tokens" columns."""
    if not results:
        return {"average_tokens_per_response": None, "mean_total_tokens_per_debate": None, "total_tokens": 0}

    per_response = []
    per_debate_totals = []
    for debate in results:
        per_debate_totals.append(debate.get("total_tokens", 0))
        for agent_hist in debate["agent_records"]:
            for rec in agent_hist:
                if rec.get("n_tokens") is not None:
                    per_response.append(rec["n_tokens"])

    return {
        "average_tokens_per_response": sum(per_response) / len(per_response) if per_response else None,
        "mean_total_tokens_per_debate": sum(per_debate_totals) / len(per_debate_totals) if per_debate_totals else None,
        "total_tokens": sum(per_debate_totals),
    }


def compute_communication_stats(results: list) -> dict:
    """
    Communication-topology stats, per the base paper's Table III (sparsity)
    and this project's checklist ("Selected communication edges",
    "Communication sparsity", "Candidate edges"). Only meaningful for
    dynamic-topology variants (dig/digra/digra_rag/digra_rag_memory) whose
    agent_records carry `partners_selected` — Standard MAD has no such
    field (fully connected every round, sparsity always 1.0 by
    definition) and should not be passed through this function.

    Candidate edges per selection event = n_agents - 1 (every OTHER agent
    is a candidate). Selected edges = len(partners_selected) for that
    event. Sparsity = selected / candidate, averaged over every
    selection event across every debate and round (round 1 excluded —
    it's seeded, not selected).

    Also reports the mean/min/max winning selection score
    (`igr_score` — raw IG for variant="dig", IGR for digra*), so DIG vs
    DIGRA score distributions can be compared directly even though they're
    on different scales (IG is unbounded by entropy normalization; IGR is
    the alpha-shifted ratio).
    """
    if not results:
        return {
            "n_agents": None, "candidate_edges": None,
            "mean_selected_edges": None, "sparsity": None,
            "n_selection_events": 0,
            "mean_selection_score": None, "min_selection_score": None, "max_selection_score": None,
        }

    n_agents = results[0]["n_agents"]
    candidate_edges = n_agents - 1

    selected_counts = []
    scores = []
    for debate in results:
        for agent_hist in debate["agent_records"]:
            for rec in agent_hist:
                if rec.get("partners_selected") is not None:
                    selected_counts.append(len(rec["partners_selected"]))
                if rec.get("igr_score") is not None:
                    scores.append(rec["igr_score"])

    mean_selected = sum(selected_counts) / len(selected_counts) if selected_counts else None
    sparsity = mean_selected / candidate_edges if (mean_selected is not None and candidate_edges) else None

    return {
        "n_agents": n_agents,
        "candidate_edges": candidate_edges,
        "mean_selected_edges": mean_selected,
        "sparsity": sparsity,
        "n_selection_events": len(selected_counts),
        "mean_selection_score": sum(scores) / len(scores) if scores else None,
        "min_selection_score": min(scores) if scores else None,
        "max_selection_score": max(scores) if scores else None,
    }


def compute_information_flow(results: list) -> dict:
    """
    Correct<->wrong information-flow counts (base paper's Fig. 6(a) /
    this project's checklist: "Correct -> Correct", "Correct -> Wrong",
    "Wrong -> Correct", "Wrong -> Wrong"). For each agent i at round t>=2
    who selected partners J, classify the flow using i's OWN
    correctness transition (was i correct at t-1, is i correct at t) as
    a proxy for what i absorbed from J — the saved records don't retain
    per-partner attribution, only the aggregate effect on the receiving
    agent, which is what the base paper's Fig. 6(a) itself reports
    (relative inflow of misleading/corrective signal, not a per-edge
    causal trace).
    """
    counts = {"correct_to_correct": 0, "correct_to_wrong": 0, "wrong_to_correct": 0, "wrong_to_wrong": 0}
    for debate in results:
        for agent_hist in debate["agent_records"]:
            by_round = {rec["round_idx"]: rec for rec in agent_hist}
            for round_idx, rec in by_round.items():
                if round_idx < 2 or rec.get("partners_selected") is None:
                    continue
                prev = by_round.get(round_idx - 1)
                if prev is None:
                    continue
                key = ("correct" if prev["is_correct"] else "wrong") + "_to_" + \
                      ("correct" if rec["is_correct"] else "wrong")
                counts[key] += 1
    total = sum(counts.values())
    ratios = {f"{k}_ratio": (v / total if total else None) for k, v in counts.items()}
    return {**counts, **ratios, "total": total}


def compute_cost_stats(results: list) -> dict:
    """Mean/total generate and forced_decode call counts per debate."""
    if not results:
        return {
            "mean_generate_calls": None, "mean_forced_decode_calls": None,
            "total_generate_calls": 0, "total_forced_decode_calls": 0, "n_debates": 0,
        }
    gens = [d["n_generate_calls"] for d in results]
    fds = [d["n_forced_decode_calls"] for d in results]
    return {
        "mean_generate_calls": sum(gens) / len(gens),
        "mean_forced_decode_calls": sum(fds) / len(fds),
        "total_generate_calls": sum(gens),
        "total_forced_decode_calls": sum(fds),
        "n_debates": len(results),
    }
