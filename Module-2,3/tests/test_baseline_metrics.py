import pytest

from src.baselines.cot_sc import run_cot_sc
from src.baselines.orchestration import _result_to_record
from src.llm.fake_client import FakeLLMClient
from src.metrics.baseline_metrics import aggregate_baseline_records


def _record(question_id, gold, alternatives, dataset, responses, n_samples_max, sc_ks):
    llm = FakeLLMClient(scripted_responses=responses)
    result = run_cot_sc(
        llm, question_id, dataset, "llama", "some question?",
        gold_answer=gold, gold_answer_alternatives=alternatives, n_samples_max=n_samples_max,
    )
    return _result_to_record(result, sc_ks)


def test_aggregate_accuracy_matches_majority_vote_fraction():
    # Question 1: unanimous correct ("Yale"). Question 2: unanimous wrong ("Yale" vs gold "Duke").
    r1 = _record("q1", "Yale", [], "nq", ["Final answer: Yale"] * 5, 5, [5])
    r2 = _record("q2", "Duke", [], "nq", ["Final answer: Yale"] * 5, 5, [5])

    agg = aggregate_baseline_records([r1, r2], k=5)
    assert agg["n_questions"] == 2
    assert agg["accuracy"] == pytest.approx(0.5)
    assert agg["dataset"] == "nq"
    assert agg["model"] == "llama"
    assert agg["n_samples"] == 5


def test_aggregate_em_f1_computed_against_majority_answer():
    r1 = _record("q1", "Yale University", [], "nq", ["Final answer: Yale University"] * 3, 3, [3])
    agg = aggregate_baseline_records([r1], k=3)
    assert agg["em"] == pytest.approx(1.0)
    assert agg["f1"] == pytest.approx(1.0)


def test_aggregate_truthfulqa_metric_is_none_for_non_truthfulqa_dataset():
    r1 = _record("q1", "Yale", [], "nq", ["Final answer: Yale"] * 3, 3, [3])
    agg = aggregate_baseline_records([r1], k=3)
    assert agg["truthfulqa_metric"] is None


def test_aggregate_truthfulqa_metric_equals_accuracy_for_truthfulqa_dataset():
    r1 = _record("q1", "Yale", [], "truthfulqa", ["Final answer: Yale"] * 3, 3, [3])
    agg = aggregate_baseline_records([r1], k=3)
    assert agg["truthfulqa_metric"] == agg["accuracy"]


def test_aggregate_wrong_consensus_rate_only_counts_genuine_majority():
    # q1: 3/3 agree on wrong answer -> genuine majority, wrong -> counts.
    # q2: split 1/1/1 (Yale/Duke/Cornell) with gold "Yale" -> majority
    #     share is 1/3, NOT a genuine majority (<=0.5) -> excluded from WCR denom.
    r1 = _record("q1", "Duke", [], "nq", ["Final answer: Cornell"] * 3, 3, [3])
    r2 = _record("q2", "Yale", [], "nq",
                 ["Final answer: Yale", "Final answer: Duke", "Final answer: Cornell"], 3, [3])
    agg = aggregate_baseline_records([r1, r2], k=3)
    assert agg["wrong_consensus_rate"] == pytest.approx(1.0)  # only q1 counted, and it's wrong


def test_aggregate_wrong_consensus_rate_none_when_no_genuine_majority():
    r1 = _record("q1", "Yale", [], "nq",
                 ["Final answer: Yale", "Final answer: Duke", "Final answer: Cornell"], 3, [3])
    agg = aggregate_baseline_records([r1], k=3)
    assert agg["wrong_consensus_rate"] is None


def test_aggregate_invalid_answer_rate_is_sample_level():
    # 1 invalid out of 3 samples for the one question -> 1/3.
    r1 = _record("q1", "Yale", [], "nq",
                 ["Final answer: ", "Final answer: Yale", "Final answer: Yale"], 3, [3])
    agg = aggregate_baseline_records([r1], k=3)
    assert agg["invalid_answer_rate"] == pytest.approx(1 / 3)


def test_aggregate_tokens_are_none_when_logprobs_not_requested():
    r1 = _record("q1", "Yale", [], "nq", ["Final answer: Yale"] * 3, 3, [3])
    agg = aggregate_baseline_records([r1], k=3)
    assert agg["average_tokens"] is None
    assert agg["total_tokens"] is None
    assert "_note" in agg


def test_aggregate_average_latency_is_present_and_nonnegative():
    r1 = _record("q1", "Yale", [], "nq", ["Final answer: Yale"] * 3, 3, [3])
    agg = aggregate_baseline_records([r1], k=3)
    assert agg["average_latency_seconds"] >= 0


def test_aggregate_returns_none_for_unrequested_k():
    r1 = _record("q1", "Yale", [], "nq", ["Final answer: Yale"] * 3, 3, [3])
    assert aggregate_baseline_records([r1], k=10) is None
