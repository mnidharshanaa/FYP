import json

import pytest

from src.llm.fake_client import FakeLLMClient
from src.reliability.entailment import (
    EntailmentMatrix,
    _parse_clustering_output,
    build_clusters,
    compute_agent_reliability,
    run_entailment_clustering,
    select_representative,
)


def _matrix_json(rows):
    return json.dumps({"matrix": rows})


# ---------------------------------------------------------------------------
# _parse_clustering_output
# ---------------------------------------------------------------------------

def test_parse_valid_matrix():
    raw = _matrix_json([[1.0, 0.9], [0.85, 1.0]])
    matrix, status = _parse_clustering_output(raw, n=2)
    assert status == "ok"
    assert matrix == [[1.0, 0.9], [0.85, 1.0]]


def test_parse_tolerates_surrounding_chatter():
    raw = "Sure, here you go:\n" + _matrix_json([[1.0, 0.0], [0.0, 1.0]]) + "\nHope that helps!"
    matrix, status = _parse_clustering_output(raw, n=2)
    assert status == "ok"
    assert matrix == [[1.0, 0.0], [0.0, 1.0]]


def test_parse_clips_out_of_range_values():
    raw = _matrix_json([[1.5, -0.2], [0.5, 1.0]])
    matrix, status = _parse_clustering_output(raw, n=2)
    assert status == "ok"
    assert matrix == [[1.0, 0.0], [0.5, 1.0]]


def test_parse_falls_back_on_invalid_json():
    matrix, status = _parse_clustering_output("not json at all", n=3)
    assert status == "failed_fallback"
    assert matrix == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def test_parse_falls_back_on_wrong_shape():
    raw = _matrix_json([[1.0, 0.5]])  # 1x2, expected 2x2
    matrix, status = _parse_clustering_output(raw, n=2)
    assert status == "failed_fallback"


def test_parse_falls_back_on_wrong_row_length():
    raw = _matrix_json([[1.0, 0.5, 0.3], [0.5, 1.0]])  # ragged
    matrix, status = _parse_clustering_output(raw, n=2)
    assert status == "failed_fallback"


def test_fallback_matrix_never_looks_like_false_consensus():
    matrix, status = _parse_clustering_output("garbage", n=4)
    clusters = build_clusters(
        EntailmentMatrix(n=4, forward_matrix=matrix, raw_llm_output="garbage", parser_status=status),
        theta=0.5,
    )
    # Fallback must yield n singleton clusters — never a false majority.
    assert len(clusters) == 4
    assert all(len(c) == 1 for c in clusters)


# ---------------------------------------------------------------------------
# build_clusters
# ---------------------------------------------------------------------------

def test_build_clusters_groups_via_transitivity():
    # 0<->1 strong, 1<->2 strong, 0<->2 weak: still one cluster via transitivity.
    m = EntailmentMatrix(n=3, forward_matrix=[
        [1.0, 0.95, 0.3],
        [0.95, 1.0, 0.9],
        [0.3, 0.9, 1.0],
    ], raw_llm_output="", parser_status="ok")
    clusters = build_clusters(m, theta=0.8)
    assert sorted(clusters) == [[0, 1, 2]]


def test_build_clusters_bidirectional_requires_both_directions():
    # F(0,1)=0.9 but F(1,0)=0.1 -> bidirectional = min(0.9,0.1) = 0.1 -> below theta -> separate clusters.
    m = EntailmentMatrix(n=2, forward_matrix=[[1.0, 0.9], [0.1, 1.0]], raw_llm_output="", parser_status="ok")
    clusters = build_clusters(m, theta=0.5)
    assert sorted(clusters) == [[0], [1]]


def test_build_clusters_same_matrix_different_theta_gives_different_clusters():
    m = EntailmentMatrix(n=3, forward_matrix=[
        [1.0, 0.7, 0.2],
        [0.7, 1.0, 0.2],
        [0.2, 0.2, 1.0],
    ], raw_llm_output="", parser_status="ok")
    loose = build_clusters(m, theta=0.5)
    strict = build_clusters(m, theta=0.9)
    assert sorted(loose) == [[0, 1], [2]]
    assert sorted(strict) == [[0], [1], [2]]  # no re-run needed to see this


# ---------------------------------------------------------------------------
# select_representative
# ---------------------------------------------------------------------------

def test_select_representative_picks_most_central_member():
    # index 1 has the highest average bidirectional score to the rest of the cluster.
    m = EntailmentMatrix(n=3, forward_matrix=[
        [1.0, 0.6, 0.5],
        [0.9, 1.0, 0.95],
        [0.5, 0.95, 1.0],
    ], raw_llm_output="", parser_status="ok")
    assert select_representative([0, 1, 2], m) == 1


def test_select_representative_singleton_cluster():
    m = EntailmentMatrix(n=1, forward_matrix=[[1.0]], raw_llm_output="", parser_status="ok")
    assert select_representative([0], m) == 0


# ---------------------------------------------------------------------------
# compute_agent_reliability
# ---------------------------------------------------------------------------

def test_compute_agent_reliability_full_consensus_gives_rho_one():
    responses = ["Final answer: Yale"] * 4
    m = EntailmentMatrix(n=4, forward_matrix=[[1.0] * 4 for _ in range(4)], raw_llm_output="", parser_status="ok")
    rec = compute_agent_reliability("q1", 1, 0, responses, m, theta=0.9, gold_answer="Yale")
    assert rec.rho == 1.0
    assert rec.r == 1.0
    assert rec.num_clusters == 1
    assert rec.representative_correct is True


def test_compute_agent_reliability_no_consensus_gives_low_rho():
    responses = ["Final answer: Yale", "Final answer: Duke", "Final answer: Cornell", "Final answer: Brown"]
    m = EntailmentMatrix(n=4, forward_matrix=[
        [1.0 if i == j else 0.0 for j in range(4)] for i in range(4)
    ], raw_llm_output="", parser_status="ok")
    rec = compute_agent_reliability("q1", 1, 0, responses, m, theta=0.5, gold_answer="Yale")
    assert rec.rho == pytest.approx(0.25)
    assert rec.num_clusters == 4


def test_compute_agent_reliability_stores_raw_scores_not_just_r():
    responses = ["Final answer: Yale"] * 3
    m = EntailmentMatrix(n=3, forward_matrix=[
        [1.0, 0.8, 0.6], [0.8, 1.0, 0.6], [0.6, 0.6, 1.0]
    ], raw_llm_output="", parser_status="ok")
    rec = compute_agent_reliability("q1", 1, 0, responses, m, theta=0.7, gold_answer="Yale")
    # 6 off-diagonal ordered pairs for n=3
    assert len(rec.forward_entailment_scores) == 6
    assert len(rec.reverse_entailment_scores) == 6


def test_compute_agent_reliability_stores_parser_status():
    responses = ["Final answer: Yale"] * 3
    m_ok = EntailmentMatrix(n=3, forward_matrix=[[1.0] * 3 for _ in range(3)],
                             raw_llm_output="", parser_status="ok")
    rec_ok = compute_agent_reliability("q1", 1, 0, responses, m_ok, theta=0.9, gold_answer="Yale")
    assert rec_ok.parser_status == "ok"

    m_failed = EntailmentMatrix(n=3, forward_matrix=[[1.0 if i == j else 0.0 for j in range(3)] for i in range(3)],
                                 raw_llm_output="garbage", parser_status="failed_fallback")
    rec_failed = compute_agent_reliability("q1", 1, 0, responses, m_failed, theta=0.9, gold_answer="Yale")
    assert rec_failed.parser_status == "failed_fallback"


def test_compute_agent_reliability_can_be_recomputed_at_different_theta_with_same_matrix():
    responses = ["Final answer: Yale"] * 3
    m = EntailmentMatrix(n=3, forward_matrix=[
        [1.0, 0.7, 0.2], [0.7, 1.0, 0.2], [0.2, 0.2, 1.0]
    ], raw_llm_output="", parser_status="ok")
    rec_loose = compute_agent_reliability("q1", 1, 0, responses, m, theta=0.5, gold_answer="Yale")
    rec_strict = compute_agent_reliability("q1", 1, 0, responses, m, theta=0.9, gold_answer="Yale")
    assert rec_loose.rho > rec_strict.rho  # same matrix, no LLM call, different theta -> different R


# ---------------------------------------------------------------------------
# run_entailment_clustering (end-to-end via FakeLLMClient)
# ---------------------------------------------------------------------------

def test_run_entailment_clustering_calls_generate_twice():
    # 1 resample call (n=3) + 1 clustering call (n=1)
    llm = FakeLLMClient(scripted_responses=[
        "Final answer: Yale", "Final answer: Yale", "Final answer: Duke",
        _matrix_json([[1.0, 0.9, 0.1], [0.9, 1.0, 0.1], [0.1, 0.1, 1.0]]),
    ])
    responses, matrix, resample_latency, clustering_latency = run_entailment_clustering(
        llm, "q1", round_idx=1, agent_id=0, prompt="Who won?", question="Who won?", n=3,
    )
    assert len(llm.calls) == 2
    assert llm.calls[0]["n"] == 3
    assert llm.calls[1]["n"] == 1
    assert len(responses) == 3
    assert matrix.parser_status == "ok"
    assert resample_latency >= 0 and clustering_latency >= 0


def test_run_entailment_clustering_end_to_end_with_theta_sweep():
    llm = FakeLLMClient(scripted_responses=[
        "Final answer: Yale", "Final answer: Yale", "Final answer: Yale", "Final answer: Duke",
        _matrix_json([
            [1.0, 0.95, 0.9, 0.1],
            [0.95, 1.0, 0.9, 0.1],
            [0.9, 0.9, 1.0, 0.1],
            [0.1, 0.1, 0.1, 1.0],
        ]),
    ])
    responses, matrix, _, _ = run_entailment_clustering(
        llm, "q1", round_idx=2, agent_id=1, prompt="Who won?", question="Who won?", n=4,
    )
    rec_92 = compute_agent_reliability("q1", 2, 1, responses, matrix, theta=0.92, gold_answer="Yale")
    rec_66 = compute_agent_reliability("q1", 2, 1, responses, matrix, theta=0.66, gold_answer="Yale")
    assert rec_92.rho == pytest.approx(2 / 4)   # only the 0.95 pair clears 0.92
    assert rec_66.rho == pytest.approx(3 / 4)   # 0.9/0.95 pairs all clear 0.66
    assert llm.calls == llm.calls  # still exactly the 2 calls made above; no extra generate() needed
    assert len(llm.calls) == 2


def test_run_entailment_clustering_raises_if_resample_count_violated():
    class BrokenClient(FakeLLMClient):
        def generate(self, *args, **kwargs):
            results = super().generate(*args, **kwargs)
            return results[:-1] if kwargs.get("n", 1) > 1 else results

    llm = BrokenClient(scripted_responses=["a", "b", "c", _matrix_json([[1.0]])])
    with pytest.raises(RuntimeError):
        run_entailment_clustering(llm, "q1", 1, 0, "prompt", "question", n=3)
