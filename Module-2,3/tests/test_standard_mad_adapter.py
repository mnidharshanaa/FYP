from src.metrics.propagation_metrics import compute_final_accuracy, compute_propagation_metrics
from src.metrics.standard_mad_adapter import adapt_standard_mad_debate, adapt_standard_mad_debates


def _debate(agent_responses, gold="Yale", alternatives=None):
    return {
        "question_id": "q1", "setup": "standard", "n_agents": len(agent_responses),
        "n_rounds": len(agent_responses[0]), "agent_responses": agent_responses,
        "gold_answer": gold, "gold_answer_alternatives": alternatives or [],
    }


def test_adapts_round_indices_to_1_indexed():
    debate = _debate([["Final answer: Yale", "Final answer: Yale"]])
    adapted = adapt_standard_mad_debate(debate)
    rounds = [r["round_idx"] for r in adapted["agent_records"][0]]
    assert rounds == [1, 2]


def test_adapts_correctness_using_gold_answer():
    debate = _debate([["Final answer: Yale", "Final answer: Duke"]], gold="Yale")
    adapted = adapt_standard_mad_debate(debate)
    records = adapted["agent_records"][0]
    assert records[0]["is_correct"] is True
    assert records[1]["is_correct"] is False


def test_adapts_multiple_agents():
    debate = _debate([
        ["Final answer: Yale"], ["Final answer: Duke"],
    ], gold="Yale")
    adapted = adapt_standard_mad_debate(debate)
    assert len(adapted["agent_records"]) == 2
    assert adapted["agent_records"][0][0]["is_correct"] is True
    assert adapted["agent_records"][1][0]["is_correct"] is False


def test_adapted_data_works_with_existing_propagation_metrics():
    # this is the real point: adapted Standard MAD data plugs directly
    # into the SAME tested MA/MR/IMR/CR implementation DIGRA data uses
    debate = _debate([
        ["Final answer: Yale", "Final answer: Duke"],  # correct -> incorrect (misled)
        ["Final answer: Duke", "Final answer: Yale"],  # incorrect -> correct (corrected)
    ], gold="Yale")
    adapted = adapt_standard_mad_debate(debate)
    metrics = compute_propagation_metrics([adapted])
    assert metrics["MA"][1] == 0.5   # 1 of 2 correct at round 1
    assert metrics["MR"][2] == 1.0   # the one correct agent became incorrect
    assert metrics["CR"][2] == 1.0   # the one incorrect agent became correct


def test_adapted_data_works_with_final_accuracy():
    debate = _debate([
        ["Final answer: Yale"], ["Final answer: Yale"], ["Final answer: Duke"],
    ], gold="Yale")
    adapted = adapt_standard_mad_debate(debate)
    acc = compute_final_accuracy([adapted])
    assert acc == 1.0  # majority (2/3) said Yale


def test_adapt_standard_mad_debates_batch():
    debates = [
        _debate([["Final answer: Yale"]], gold="Yale"),
        _debate([["Final answer: Duke"]], gold="Yale"),
    ]
    adapted = adapt_standard_mad_debates(debates)
    assert len(adapted) == 2
    assert adapted[0]["agent_records"][0][0]["is_correct"] is True
    assert adapted[1]["agent_records"][0][0]["is_correct"] is False
