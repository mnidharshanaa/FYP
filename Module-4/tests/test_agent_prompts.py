from src.agents.prompts import build_debate_round_prompt


def test_prompt_contains_question_and_own_response():
    p = build_debate_round_prompt(
        question="Who won?", own_previous_response="I said Duke.", other_responses=["I said Yale."]
    )
    assert "Who won?" in p
    assert "I said Duke." in p
    assert "I said Yale." in p


def test_prompt_numbers_multiple_other_agents():
    p = build_debate_round_prompt(
        question="Q", own_previous_response="own",
        other_responses=["resp A", "resp B", "resp C"],
    )
    assert "Agent 1's response:" in p
    assert "Agent 2's response:" in p
    assert "Agent 3's response:" in p
    assert "resp A" in p and "resp B" in p and "resp C" in p


def test_prompt_handles_no_other_agents():
    p = build_debate_round_prompt(question="Q", own_previous_response="own", other_responses=[])
    assert "no other agents" in p


def test_prompt_ends_with_final_answer_instruction():
    p = build_debate_round_prompt(question="Q", own_previous_response="own", other_responses=["x"])
    assert "Final answer:" in p
