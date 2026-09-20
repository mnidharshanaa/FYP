import pytest

from src.reports.propagation_report import (
    four_quadrant_table,
    lambda_ablation_table,
    rag_effect_summary,
)


def _comm(source_agent, target_agent, quadrant, pre_rag_decision, information_flow=None,
          result_correct=None, lam=0.5):
    return {
        "question_id": "q1", "round_idx": 1, "source_agent": source_agent, "target_agent": target_agent,
        "candidate_edge": True, "selected_edge": True, "ig": None, "entropy": 0.5,
        "igr": 0.5, "u": 0.5, "r_source": 0.5, "lam": lam, "ps": 0.5,
        "propagation_threshold": 0.6, "utility_threshold": 0.5,
        "decision": "propagate", "pre_rag_decision": pre_rag_decision, "quadrant": quadrant,
        "origin_correct": True, "result_correct": result_correct, "information_flow": information_flow,
    }


def _agent_round(round_idx, is_correct, extracted_answer="Yale"):
    return {"round_idx": round_idx, "response_text": f"Final answer: {extracted_answer}",
            "extracted_answer": extracted_answer, "is_correct": is_correct, "entropy": 0.5}


def _debate(question_id, comms, agent_records, gold_answer="Yale", total_tokens=100):
    return {
        "question_id": question_id, "setup": "standard", "n_agents": len(agent_records),
        "n_rounds_requested": 2, "n_rounds_run": 2, "early_stopped": False,
        "gold_answer": gold_answer, "gold_answer_alternatives": [],
        "agent_records": agent_records, "agent_reliability_records": [],
        "communication_records": comms, "rag_verification_records": [],
        "n_generate_calls": 5, "n_forced_decode_calls": 3, "n_rag_calls": 0,
        "total_tokens_partial": total_tokens,
    }


def _rag_record(source_agent, evidence_status, decision_after, r_before=0.3, r_after=0.7,
                 ps_before=0.4, ps_after=0.8):
    return {
        "question_id": "q1", "round_idx": 1, "source_agent": source_agent, "target_agent": None,
        "retrieval_query": "claim", "document_ids": ["d1"], "document_ranks": [1],
        "retrieval_scores": [1.0], "evidence_support_score": 0.9 if evidence_status == "ok" else None,
        "evidence_status": evidence_status, "r_before": r_before, "r_after": r_after,
        "ps_before": ps_before, "ps_after": ps_after, "decision_before": "verify",
        "decision_after": decision_after, "rag_tokens": None, "rag_latency_seconds": None,
        "retriever_name": "SnapshotRetriever",
    }


# ---------------------------------------------------------------------------
# lambda_ablation_table
# ---------------------------------------------------------------------------

def test_lambda_ablation_groups_by_lambda_from_data_not_filename():
    agent_records = [[_agent_round(1, True), _agent_round(2, True)]] * 3
    debate_lam0 = _debate("q1", [_comm(0, 1, "propagate", "propagate", lam=0.0)], agent_records)
    debate_lam1 = _debate("q2", [_comm(0, 1, "propagate", "propagate", lam=1.0)], agent_records)

    rows = lambda_ablation_table([debate_lam0, debate_lam1])
    assert [r["lambda"] for r in rows] == [0.0, 1.0]


def test_lambda_ablation_computes_accuracy_and_flow_rates():
    agent_records_all_correct = [[_agent_round(1, True), _agent_round(2, True)]] * 3
    comms = [
        _comm(0, 1, "propagate", "propagate", information_flow="C->C", result_correct=True, lam=0.5),
        _comm(2, 1, "verify", "verify", information_flow="W->C", result_correct=True, lam=0.5),
    ]
    debate = _debate("q1", comms, agent_records_all_correct)
    rows = lambda_ablation_table([debate])
    assert len(rows) == 1
    row = rows[0]
    assert row["lambda"] == 0.5
    assert row["n_debates"] == 1
    assert row["accuracy"] == 1.0
    assert row["WCR"] == pytest.approx(1.0)  # the one wrong-origin edge got corrected
    assert row["rag_trigger_rate"] == pytest.approx(0.5)  # 1 of 2 edges pre_rag_decision=="verify"
    assert row["avg_tokens_partial"] == 100


def test_lambda_ablation_empty_input_returns_empty_list():
    assert lambda_ablation_table([]) == []


# ---------------------------------------------------------------------------
# four_quadrant_table
# ---------------------------------------------------------------------------

def test_four_quadrant_table_groups_and_orders_correctly():
    comms = [
        _comm(0, 1, "propagate", "propagate", information_flow="C->C", result_correct=True),
        _comm(1, 0, "suppress", "suppress", information_flow="W->W", result_correct=False),
        _comm(2, 0, "verify", "verify", information_flow="W->C", result_correct=True),
    ]
    debate = _debate("q1", comms, [[_agent_round(1, True)]] * 3)
    rows = four_quadrant_table([debate])
    quadrants = [r["quadrant"] for r in rows]
    assert quadrants == ["propagate", "verify", "suppress"]  # canonical order, only populated ones


def test_four_quadrant_table_computes_per_quadrant_accuracy_and_harm():
    comms = [
        _comm(0, 1, "suppress", "suppress", information_flow="W->W", result_correct=False),
        _comm(0, 2, "suppress", "suppress", information_flow="C->W", result_correct=False),
    ]
    debate = _debate("q1", comms, [[_agent_round(1, True)]] * 3)
    rows = four_quadrant_table([debate])
    suppress_row = next(r for r in rows if r["quadrant"] == "suppress")
    assert suppress_row["n_edges"] == 2
    assert suppress_row["accuracy_after_propagation"] == 0.0
    assert suppress_row["harmful_propagation_rate"] == 1.0  # both are dangerous flow types


def test_four_quadrant_table_omits_quadrants_with_zero_edges():
    comms = [_comm(0, 1, "propagate", "propagate", information_flow="C->C", result_correct=True)]
    debate = _debate("q1", comms, [[_agent_round(1, True)]] * 3)
    rows = four_quadrant_table([debate])
    assert len(rows) == 1
    assert rows[0]["quadrant"] == "propagate"


# ---------------------------------------------------------------------------
# rag_effect_summary
# ---------------------------------------------------------------------------

def test_rag_effect_summary_none_when_no_rag_events():
    debate = _debate("q1", [_comm(0, 1, "propagate", "propagate")], [[_agent_round(1, True)]] * 3)
    assert rag_effect_summary([debate]) is None


def test_rag_effect_summary_computes_rates_and_deltas():
    debate = {
        **_debate("q1", [
            _comm(0, 1, "verify", "verify"),
            _comm(2, 1, "verify", "verify"),
        ], [[_agent_round(1, True)]] * 3),
        "rag_verification_records": [
            _rag_record(0, "ok", "propagate", r_before=0.3, r_after=0.8, ps_before=0.4, ps_after=0.9),
            _rag_record(2, "ok", "suppress", r_before=0.3, r_after=0.2, ps_before=0.4, ps_after=0.3),
        ],
    }
    summary = rag_effect_summary([debate])
    assert summary["n_verify_triggering_edges"] == 2
    assert summary["n_rag_events_deduplicated"] == 2
    assert summary["n_completed"] == 2
    assert summary["verify_to_propagate_rate"] == pytest.approx(0.5)
    assert summary["verify_to_suppress_rate"] == pytest.approx(0.5)
    assert summary["mean_delta_r"] == pytest.approx(((0.8 - 0.3) + (0.2 - 0.3)) / 2)


def test_rag_effect_summary_tracks_no_evidence_and_parse_failures_separately():
    debate = {
        **_debate("q1", [_comm(0, 1, "verify", "verify")], [[_agent_round(1, True)]] * 3),
        "rag_verification_records": [_rag_record(0, "no_evidence_available", "suppress")],
    }
    summary = rag_effect_summary([debate])
    assert summary["n_no_evidence_available"] == 1
    assert summary["n_completed"] == 0
    assert summary["mean_delta_r"] is None  # never computed from a non-completed event
