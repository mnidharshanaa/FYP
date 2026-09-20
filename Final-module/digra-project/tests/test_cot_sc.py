import math

import pytest

from src.baselines.cot_sc import majority_vote, run_cot_sc
from src.llm.fake_client import FakeLLMClient


def _fake(responses):
    return FakeLLMClient(scripted_responses=responses)


def test_run_cot_sc_calls_generate_once_with_n_samples_max():
    llm = _fake(["Reasoning...\nFinal answer: Yale"] * 5)
    result = run_cot_sc(
        llm, "q1", "nq", "llama", "Who won?", gold_answer="Yale", n_samples_max=5,
    )
    assert len(llm.calls) == 1
    assert llm.calls[0]["method"] == "generate"
    assert llm.calls[0]["n"] == 5
    assert len(result.samples) == 5


def test_run_cot_sc_grades_each_sample_independently():
    llm = _fake([
        "Final answer: Yale",
        "Final answer: Duke",
        "Final answer: Yale",
    ])
    result = run_cot_sc(llm, "q1", "nq", "llama", "Who won?", gold_answer="Yale", n_samples_max=3)
    correctness = [s.is_correct for s in result.samples]
    assert correctness == [True, False, True]


def test_run_cot_sc_marks_unparseable_answer_invalid():
    llm = _fake(["I'm not sure about this one."])
    result = run_cot_sc(llm, "q1", "nq", "llama", "Who won?", gold_answer="Yale", n_samples_max=1)
    # extract_final_answer falls back to the full text when no marker is
    # found, so this is "invalid" only in the sense that grading.is_correct
    # will (correctly) mark it wrong — the real invalid case is an empty
    # response after extraction, tested below.
    assert result.samples[0].is_correct is False


def test_run_cot_sc_raises_if_backend_violates_n_contract():
    class BrokenClient(FakeLLMClient):
        def generate(self, *args, **kwargs):
            return super().generate(*args, **kwargs)[:-1]  # drop one result

    llm = BrokenClient(scripted_responses=["Final answer: Yale"] * 5)
    with pytest.raises(RuntimeError):
        run_cot_sc(llm, "q1", "nq", "llama", "Who won?", gold_answer="Yale", n_samples_max=5)


# ---------------------------------------------------------------------------
# majority_vote
# ---------------------------------------------------------------------------

def test_majority_vote_k1_is_plain_cot():
    llm = _fake(["Final answer: Yale", "Final answer: Duke"])
    result = run_cot_sc(llm, "q1", "nq", "llama", "Who won?", gold_answer="Yale", n_samples_max=2)
    vote = majority_vote(result, k=1)
    assert vote.majority_answer.lower() == "yale"
    assert vote.majority_is_correct is True
    assert vote.majority_vote_share == 1.0
    assert vote.vote_entropy == 0.0
    assert vote.n_unique_answers == 1


def test_majority_vote_clear_majority():
    llm = _fake([
        "Final answer: Yale",
        "Final answer: Yale",
        "Final answer: Yale",
        "Final answer: Duke",
        "Final answer: Cornell",
    ])
    result = run_cot_sc(llm, "q1", "nq", "llama", "Who won?", gold_answer="Yale", n_samples_max=5)
    vote = majority_vote(result, k=5)
    assert vote.majority_answer.lower() == "yale"
    assert vote.majority_is_correct is True
    assert vote.majority_vote_share == pytest.approx(3 / 5)
    assert vote.n_unique_answers == 3
    assert vote.vote_entropy > 0.0


def test_majority_vote_wrong_consensus():
    llm = _fake(["Final answer: Duke"] * 5)
    result = run_cot_sc(llm, "q1", "nq", "llama", "Who won?", gold_answer="Yale", n_samples_max=5)
    vote = majority_vote(result, k=5)
    assert vote.majority_answer.lower() == "duke"
    assert vote.majority_is_correct is False
    assert vote.majority_vote_share == 1.0


def test_majority_vote_slices_to_first_k_only():
    # First 3 are unanimous "Yale"; last 2 are "Duke" — with k=3 the vote
    # must be computed from only the first 3 samples, ignoring the rest.
    llm = _fake([
        "Final answer: Yale",
        "Final answer: Yale",
        "Final answer: Yale",
        "Final answer: Duke",
        "Final answer: Duke",
    ])
    result = run_cot_sc(llm, "q1", "nq", "llama", "Who won?", gold_answer="Yale", n_samples_max=5)
    vote_k3 = majority_vote(result, k=3)
    assert vote_k3.majority_answer.lower() == "yale"
    assert vote_k3.majority_vote_share == 1.0


def test_majority_vote_invalid_answers_tracked():
    llm = _fake([
        "Final answer: ",       # extracts to empty string -> invalid
        "Final answer: Yale",
        "Final answer: Yale",
    ])
    result = run_cot_sc(llm, "q1", "nq", "llama", "Who won?", gold_answer="Yale", n_samples_max=3)
    vote = majority_vote(result, k=3)
    assert vote.n_invalid == 1
    assert vote.invalid_rate == pytest.approx(1 / 3)
    # the majority answer should still be the valid "Yale" cluster
    assert vote.majority_answer.lower() == "yale"


def test_majority_vote_out_of_range_k_raises():
    llm = _fake(["Final answer: Yale"] * 3)
    result = run_cot_sc(llm, "q1", "nq", "llama", "Who won?", gold_answer="Yale", n_samples_max=3)
    with pytest.raises(ValueError):
        majority_vote(result, k=0)
    with pytest.raises(ValueError):
        majority_vote(result, k=4)


def test_majority_vote_entropy_matches_shannon_formula():
    # 3-way split of 1/1/1 over k=3 -> entropy = ln(3)
    llm = _fake(["Final answer: Yale", "Final answer: Duke", "Final answer: Cornell"])
    result = run_cot_sc(llm, "q1", "nq", "llama", "Who won?", gold_answer="Yale", n_samples_max=3)
    vote = majority_vote(result, k=3)
    assert vote.vote_entropy == pytest.approx(math.log(3), abs=1e-9)
