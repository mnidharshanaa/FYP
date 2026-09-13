import pytest

from src.metrics.propagation_metrics import (
    compute_communication_stats,
    compute_cost_stats,
    compute_em_f1,
    compute_final_accuracy,
    compute_ig_per_round,
    compute_information_flow,
    compute_majority_vote_stats,
    compute_propagation_metrics,
    compute_rag_precision_recall,
    compute_token_stats,
)


def _agent_record(round_idx, correct, entropy=0.5, rag_flagged=None, answer="Yale",
                   partners_selected=None, igr_score=None, candidate_scores=None):
    return {
        "round_idx": round_idx, "response_text": "x", "extracted_answer": answer,
        "is_correct": correct, "entropy": entropy, "rag_flagged": rag_flagged,
        "cross_round_inconsistent": None, "combined_flagged": None,
        "trust_after_round": None, "partners_selected": partners_selected, "igr_score": igr_score,
        "candidate_scores": candidate_scores, "n_tokens": None,
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


# ---------------------------------------------------------------------------
# compute_communication_stats
# ---------------------------------------------------------------------------

def test_communication_stats_basic():
    debate = _debate([
        [_agent_record(1, True), _agent_record(2, True, partners_selected=[1], igr_score=0.4)],
        [_agent_record(1, True), _agent_record(2, True, partners_selected=[0], igr_score=0.3)],
        [_agent_record(1, False), _agent_record(2, False, partners_selected=[0, 1], igr_score=0.6)],
    ])
    stats = compute_communication_stats([debate])
    assert stats["n_agents"] == 3
    assert stats["candidate_edges"] == 2
    assert stats["n_selection_events"] == 3
    assert stats["mean_selected_edges"] == pytest.approx((1 + 1 + 2) / 3)
    assert stats["sparsity"] == pytest.approx(((1 + 1 + 2) / 3) / 2)
    assert stats["mean_selection_score"] == pytest.approx((0.4 + 0.3 + 0.6) / 3)


def test_communication_stats_empty_results():
    stats = compute_communication_stats([])
    assert stats["sparsity"] is None
    assert stats["n_selection_events"] == 0


# ---------------------------------------------------------------------------
# compute_information_flow
# ---------------------------------------------------------------------------

def test_information_flow_counts_transitions():
    # agent goes wrong (r1) -> correct (r2): wrong_to_correct
    debate = _debate([
        [_agent_record(1, False), _agent_record(2, True, partners_selected=[1])],
        # agent goes correct (r1) -> wrong (r2): correct_to_wrong
        [_agent_record(1, True), _agent_record(2, False, partners_selected=[0])],
    ])
    flow = compute_information_flow([debate])
    assert flow["wrong_to_correct"] == 1
    assert flow["correct_to_wrong"] == 1
    assert flow["total"] == 2
    assert flow["wrong_to_correct_ratio"] == 0.5
    assert flow["correct_to_wrong_ratio"] == 0.5


def test_information_flow_ignores_round_1_and_unselected_rounds():
    # round 1 is seeded, not a selection event; round without
    # partners_selected shouldn't count either
    debate = _debate([[_agent_record(1, True), _agent_record(2, True, partners_selected=None)]])
    flow = compute_information_flow([debate])
    assert flow["total"] == 0


def test_information_flow_empty_results():
    flow = compute_information_flow([])
    assert flow["total"] == 0
    assert flow["correct_to_correct_ratio"] is None


# ---------------------------------------------------------------------------
# compute_em_f1
# ---------------------------------------------------------------------------

def test_em_f1_uses_final_round_answer_per_agent():
    debate = _debate([
        [_agent_record(1, True, answer="Duke"), _agent_record(2, True, answer="Yale")],
        [_agent_record(1, False, answer="Yale"), _agent_record(2, False, answer="Harvard")],
    ])
    stats = compute_em_f1([debate])
    # final-round answers: "Yale" (agent0, matches gold "Yale") and
    # "Harvard" (agent1, doesn't match) -> EM = 1/2
    assert stats["n"] == 2
    assert stats["EM"] == 0.5
    assert stats["F1"] == 0.5  # exact-match tokens also score F1=1.0/0.0 here


def test_em_f1_empty_results():
    stats = compute_em_f1([])
    assert stats["EM"] is None
    assert stats["n"] == 0


# ---------------------------------------------------------------------------
# compute_majority_vote_stats
# ---------------------------------------------------------------------------

def test_majority_vote_stats_full_consensus():
    debate = _debate([
        [_agent_record(1, True, answer="Yale"), _agent_record(2, True, answer="Yale")],
        [_agent_record(1, True, answer="Yale"), _agent_record(2, True, answer="Yale")],
    ])
    stats = compute_majority_vote_stats([debate])
    assert stats["majority_vote_share"] == 1.0
    assert stats["vote_entropy"] == 0.0
    assert stats["mean_unique_answers"] == 1


def test_majority_vote_stats_split_vote():
    # final round: 2 agents say "Yale", 1 says "Duke" -> majority share 2/3
    debate = _debate([
        [_agent_record(1, True, answer="Yale"), _agent_record(2, True, answer="Yale")],
        [_agent_record(1, True, answer="Yale"), _agent_record(2, True, answer="Yale")],
        [_agent_record(1, False, answer="Duke"), _agent_record(2, False, answer="Duke")],
    ])
    stats = compute_majority_vote_stats([debate])
    assert stats["majority_vote_share"] == pytest.approx(2 / 3)
    assert stats["mean_unique_answers"] == 2
    assert stats["vote_entropy"] > 0.0


def test_majority_vote_stats_empty_results():
    stats = compute_majority_vote_stats([])
    assert stats["majority_vote_share"] is None
    assert stats["n_debates"] == 0


# ---------------------------------------------------------------------------
# compute_ig_per_round
# ---------------------------------------------------------------------------

def test_ig_per_round_groups_by_round_and_keeps_every_candidate():
    debate = _debate([
        [
            _agent_record(1, True),
            _agent_record(2, True, candidate_scores={"1": 0.2, "2": 0.5}),
        ],
        [
            _agent_record(1, True),
            _agent_record(2, True, candidate_scores={"0": 0.1, "2": 0.3}),
        ],
    ])
    by_round = compute_ig_per_round([debate])
    assert set(by_round.keys()) == {2}
    assert by_round[2]["n_edges"] == 4  # 2 + 2 scores, winners and losers both kept
    assert by_round[2]["max"] == 0.5
    assert by_round[2]["min"] == 0.1


def test_ig_per_round_empty_results():
    assert compute_ig_per_round([]) == {}


# ---------------------------------------------------------------------------
# compute_token_stats
# ---------------------------------------------------------------------------

def test_token_stats_basic():
    debate = _debate([[_agent_record(1, True)], [_agent_record(1, True)]])
    debate["agent_records"][0][0]["n_tokens"] = 10
    debate["agent_records"][1][0]["n_tokens"] = 20
    debate["total_tokens"] = 30
    stats = compute_token_stats([debate])
    assert stats["average_tokens_per_response"] == 15.0
    assert stats["mean_total_tokens_per_debate"] == 30.0
    assert stats["total_tokens"] == 30


def test_token_stats_empty_results():
    stats = compute_token_stats([])
    assert stats["total_tokens"] == 0
    assert stats["average_tokens_per_response"] is None
