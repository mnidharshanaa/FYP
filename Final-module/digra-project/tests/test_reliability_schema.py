import json
from dataclasses import asdict

from src.reliability.schema import (
    AgentReliabilityRecord,
    CommunicationRecord,
    RagVerificationRecord,
    RawResponseRecord,
    RunMetadata,
)


def test_raw_response_record_round_trips_through_json():
    rec = RawResponseRecord(
        question_id="nq_0000", dataset="nq", model="llama", agent_id=0, round_idx=1,
        sample_idx=0, response="Final answer: Yale", extracted_answer="Yale",
        ground_truth="Yale", is_correct=True, prompt="Who won?", seed=0,
        temperature=1.0, top_p=1.0, max_tokens=300,
    )
    reloaded = json.loads(json.dumps(asdict(rec)))
    assert reloaded["question_id"] == "nq_0000"
    assert reloaded["is_correct"] is True
    assert reloaded["total_tokens"] is None  # optional field defaults to None, not omitted


def test_agent_reliability_record_carries_raw_scores_for_post_hoc_theta_sweep():
    rec = AgentReliabilityRecord(
        question_id="nq_0000", round_idx=1, agent_id=0, n=10, theta=0.92,
        forward_entailment_scores=[0.95, 0.40, 0.88],
        reverse_entailment_scores=[0.91, 0.35, 0.90],
        num_clusters=2, largest_cluster_size=7, rho=0.7, r=0.7,
        representative_response="Final answer: Yale", representative_answer="Yale",
        representative_correct=True, entailment_model="same_llm:llama",
    )
    d = asdict(rec)
    # The raw per-pair scores must survive intact — this is what makes a
    # later theta re-sweep possible without rerunning the model.
    assert d["forward_entailment_scores"] == [0.95, 0.40, 0.88]
    assert d["reverse_entailment_scores"] == [0.91, 0.35, 0.90]
    json.dumps(d)  # must be JSON-serializable for append_jsonl


def test_communication_record_defaults_let_partial_construction_work():
    # An edge that was a candidate but never selected shouldn't require
    # every downstream field (U, R, PS, decision...) to be filled in.
    rec = CommunicationRecord(
        question_id="nq_0000", round_idx=1, source_agent=0, target_agent=1,
        candidate_edge=True, selected_edge=False,
    )
    d = asdict(rec)
    assert d["selected_edge"] is False
    assert d["decision"] is None
    json.dumps(d)


def test_rag_verification_record_before_after_fields_present():
    rec = RagVerificationRecord(
        question_id="nq_0000", round_idx=2, source_agent=0, target_agent=1,
        retrieval_query="largest planet in the solar system",
        document_ids=["wiki:Jupiter"], document_ranks=[1], retrieval_scores=[0.93],
        evidence_support_score=0.86, r_before=0.28, r_after=0.79,
        ps_before=0.505, ps_after=0.849, decision_before="verify", decision_after="propagate",
        rag_tokens=120, rag_latency_seconds=0.8, retriever_name="wikipedia_snapshot",
    )
    d = asdict(rec)
    assert d["r_before"] < d["r_after"]
    assert d["decision_before"] == "verify"
    assert d["decision_after"] == "propagate"
    json.dumps(d)


def test_run_metadata_keeps_digra_alpha_and_lambda_as_distinct_fields():
    meta = RunMetadata(
        run_id="run_001", timestamp="2026-09-19T00:00:00Z", model="llama", dataset="nq",
        method="propagation_control", n_agents=3, n_rounds=3, temperature=1.0, top_p=1.0,
        seed=0, max_tokens=300, digra_alpha=0.2, lam=0.5,
        propagation_threshold=0.6, utility_threshold=0.5, n_resample=10, theta=0.92,
        entailment_model="same_llm:llama",
    )
    d = asdict(meta)
    assert d["digra_alpha"] == 0.2
    assert d["lam"] == 0.5
    assert d["digra_alpha"] != d["lam"]
    json.dumps(d)
