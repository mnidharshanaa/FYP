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

from src.metrics.grading import normalize_answer


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
