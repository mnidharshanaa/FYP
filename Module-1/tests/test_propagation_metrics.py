from src.metrics.propagation_metrics import (
    compute_cost_stats,
    compute_final_accuracy,
    compute_propagation_metrics,
    compute_rag_precision_recall,
)


def _agent_record(round_idx, correct, entropy=0.5, rag_flagged=None, answer="Yale"):
    return {
        "round_idx": round_idx, "response_text": "x", "extracted_answer": answer,
        "is_correct": correct, "entropy": entropy, "rag_flagged": rag_flagged,
        "cross_round_inconsistent": None, "combined_flagged": None,
        "trust_after_round": None, "partners_selected": None, "igr_score": None,
    }


def _debate(agent_records, gen_calls=2, fd_calls=3):
    return {
        "question_id": "q1", "variant": "digra", "setup": "standard",
        "n_agents": len(agent_records), "n_rounds_requested": 3,
        "n_rounds_run": max(r["round_idx"] for hist in agent_records for r in hist),
        "early_stopped": False, "gold_answer": "Yale", "gold_answer_alternatives": [],
        "agent_records": agent_records,
        "n_generate_calls": gen_calls, "n_forced_decode_calls": fd_calls,
    }


def test_ma_all_correct_all_rounds():
    debate = _debate([
        [_agent_record(1, True), _agent_record(2, True)],
        [_agent_record(1, True), _agent_record(2, True)],
    ])
    metrics = compute_propagation_metrics([debate])
    assert metrics["MA"][1] == 1.0
    assert metrics["MA"][2] == 1.0


def test_mr_correct_agent_becomes_incorrect():
    debate = _debate([[_agent_record(1, True), _agent_record(2, False)]])
    metrics = compute_propagation_metrics([debate])
    assert metrics["MR"][2] == 1.0


def test_cr_incorrect_agent_becomes_correct():
    debate = _debate([[_agent_record(1, False), _agent_record(2, True)]])
    metrics = compute_propagation_metrics([debate])
    assert metrics["CR"][2] == 1.0


def test_imr_equals_mr_at_round_2():
    debate = _debate([
        [_agent_record(1, True), _agent_record(2, False)],
        [_agent_record(1, True), _agent_record(2, True)],
    ])
    metrics = compute_propagation_metrics([debate])
    assert metrics["IMR"][2] == metrics["MR"][2]


def test_max_round_detected_from_data():
    debate = _debate([[_agent_record(1, True), _agent_record(2, True), _agent_record(3, True)]])
    metrics = compute_propagation_metrics([debate])
    assert metrics["max_round"] == 3


def test_final_accuracy_majority_correct():
    debate = _debate([
        [_agent_record(1, True, answer="Yale")],
        [_agent_record(1, True, answer="Yale")],
        [_agent_record(1, False, answer="Duke")],
    ])
    assert compute_final_accuracy([debate]) == 1.0


def test_final_accuracy_majority_incorrect():
    debate = _debate([
        [_agent_record(1, False, answer="Duke")],
        [_agent_record(1, False, answer="Duke")],
        [_agent_record(1, True, answer="Yale")],
    ])
    assert compute_final_accuracy([debate]) == 0.0


def test_final_accuracy_uses_last_round_per_agent():
    debate = _debate([
        [_agent_record(1, False, answer="Duke"), _agent_record(2, True, answer="Yale")],
        [_agent_record(1, True, answer="Yale"), _agent_record(2, True, answer="Yale")],
    ])
    assert compute_final_accuracy([debate]) == 1.0


def test_final_accuracy_empty_results():
    assert compute_final_accuracy([]) is None


def test_rag_precision_recall_perfect_detector():
    debate = _debate([
        [_agent_record(1, correct=False, entropy=0.1, rag_flagged=True)],
        [_agent_record(1, correct=True, entropy=0.5, rag_flagged=False)],
    ])
    result = compute_rag_precision_recall([debate])
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0


def test_rag_precision_recall_missed_case():
    debate = _debate([[_agent_record(1, correct=False, entropy=0.1, rag_flagged=False)]])
    result = compute_rag_precision_recall([debate])
    assert result["recall"] == 0.0


def test_rag_precision_recall_none_when_no_rag_used():
    debate = _debate([[_agent_record(1, correct=True, rag_flagged=None)]])
    result = compute_rag_precision_recall([debate])
    assert result["precision"] is None
    assert result["n_evaluated"] == 0


def test_confident_wrong_recall_catches_low_entropy_incorrect():
    debate = _debate([
        [_agent_record(1, correct=False, entropy=0.01, rag_flagged=True)],
        [_agent_record(1, correct=False, entropy=0.9, rag_flagged=True)],
    ])
    result = compute_rag_precision_recall([debate], entropy_percentile=0.5)
    assert result["confident_wrong_recall"] == 1.0


def test_cost_stats_basic():
    debates = [_debate([[_agent_record(1, True)]], gen_calls=2, fd_calls=4),
               _debate([[_agent_record(1, True)]], gen_calls=4, fd_calls=8)]
    stats = compute_cost_stats(debates)
    assert stats["mean_generate_calls"] == 3.0
    assert stats["total_generate_calls"] == 6
    assert stats["n_debates"] == 2


def test_cost_stats_empty():
    stats = compute_cost_stats([])
    assert stats["n_debates"] == 0
