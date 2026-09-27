import random

import pytest

from src.agents.debate import (
    DebateResult,
    generate_round_one_standard,
    run_debate,
)
from src.llm.fake_client import FakeLLMClient

CORRECT_POOL = ["correct 1", "correct 2", "correct 3", "correct 4", "correct 5"]
INCORRECT_POOL = ["appeal 1", "appeal 2", "appeal 3"]


# ---------------------------------------------------------------------------
# generate_round_one_standard
# ---------------------------------------------------------------------------

def test_generate_round_one_standard_calls_generate_once_with_n_agents():
    llm = FakeLLMClient(scripted_responses=["r1", "r2", "r3"])
    texts = generate_round_one_standard(llm, question="Q", n_agents=3, seed=0)
    assert texts == ["r1", "r2", "r3"]
    assert len(llm.calls) == 1
    assert llm.calls[0]["n"] == 3


# ---------------------------------------------------------------------------
# run_debate — setup validation
# ---------------------------------------------------------------------------

def test_run_debate_rejects_setup_not_summing_to_n_agents():
    llm = FakeLLMClient(scripted_responses=[])
    with pytest.raises(ValueError):
        run_debate(
            llm=llm, question_id="q1", question="Q", gold_answer="Yale",
            n_agents=3, n_rounds=3, setup=(1, 1),  # sums to 2, not 3
            correct_pool=CORRECT_POOL, incorrect_pool=INCORRECT_POOL,
        )


# ---------------------------------------------------------------------------
# run_debate — seeded round 1 (no LLM call for round 1 itself)
# ---------------------------------------------------------------------------

def test_seeded_round1_makes_no_llm_call_for_round1():
    # 3 agents, 2 rounds -> round 2 is ONE batched call (generate_batch),
    # not one call per agent — see generate_batch's docstring for why.
    llm = FakeLLMClient(scripted_responses=["r2_a0", "r2_a1", "r2_a2"])
    result = run_debate(
        llm=llm, question_id="q1", question="Q", gold_answer="Yale",
        n_agents=3, n_rounds=2, setup=(2, 1),
        correct_pool=CORRECT_POOL, incorrect_pool=INCORRECT_POOL, seed=0,
    )
    assert len(llm.calls) == 1  # one batched round-2 call covering all 3 agents
    assert llm.calls[0]["method"] == "generate_batch"
    assert llm.calls[0]["n"] == 3
    assert result.setup == "2,1"


def test_seeded_round1_responses_come_from_pools():
    llm = FakeLLMClient(scripted_responses=["r2_a0", "r2_a1", "r2_a2"])
    result = run_debate(
        llm=llm, question_id="q1", question="Q", gold_answer="Yale",
        n_agents=3, n_rounds=2, setup=(2, 1),
        correct_pool=CORRECT_POOL, incorrect_pool=INCORRECT_POOL, seed=0,
    )
    round1_texts = [result.agent_responses[i][0] for i in range(3)]
    n_from_correct = sum(1 for t in round1_texts if t in CORRECT_POOL)
    n_from_incorrect = sum(1 for t in round1_texts if t in INCORRECT_POOL)
    assert n_from_correct == 1
    assert n_from_incorrect == 2


def test_upper_bound_setup_all_correct_round1():
    llm = FakeLLMClient(scripted_responses=["r2_a0", "r2_a1", "r2_a2"])
    result = run_debate(
        llm=llm, question_id="q1", question="Q", gold_answer="Yale",
        n_agents=3, n_rounds=2, setup=(0, 3),
        correct_pool=CORRECT_POOL, incorrect_pool=INCORRECT_POOL, seed=0,
    )
    round1_texts = [result.agent_responses[i][0] for i in range(3)]
    assert all(t in CORRECT_POOL for t in round1_texts)


# ---------------------------------------------------------------------------
# run_debate — standard setup (round 1 IS an LLM call)
# ---------------------------------------------------------------------------

def test_standard_setup_round1_uses_llm():
    scripted = ["fresh_a0", "fresh_a1", "fresh_a2"] + ["r2_a0", "r2_a1", "r2_a2"]
    llm = FakeLLMClient(scripted_responses=scripted)
    result = run_debate(
        llm=llm, question_id="q1", question="Q", gold_answer="Yale",
        n_agents=3, n_rounds=2, setup="standard", seed=0,
    )
    assert result.setup == "standard"
    assert [result.agent_responses[i][0] for i in range(3)] == ["fresh_a0", "fresh_a1", "fresh_a2"]
    # 1 call for round1 (n=3, same prompt) + 1 batched call for round2
    # (3 different prompts, 1 completion each) = 2 total calls
    assert len(llm.calls) == 2
    assert llm.calls[0]["method"] == "generate"
    assert llm.calls[1]["method"] == "generate_batch"


# ---------------------------------------------------------------------------
# run_debate — multi-round shape and call counts
# ---------------------------------------------------------------------------

def test_agent_responses_shape_matches_n_agents_and_n_rounds():
    llm = FakeLLMClient(response_fn=lambda p: "resp")
    result = run_debate(
        llm=llm, question_id="q1", question="Q", gold_answer="Yale",
        n_agents=3, n_rounds=3, setup=(1, 2),
        correct_pool=CORRECT_POOL, incorrect_pool=INCORRECT_POOL, seed=0,
    )
    assert len(result.agent_responses) == 3
    assert all(len(agent_hist) == 3 for agent_hist in result.agent_responses)


def test_seeded_setup_n_rounds_1_makes_zero_llm_calls():
    llm = FakeLLMClient(scripted_responses=[])
    result = run_debate(
        llm=llm, question_id="q1", question="Q", gold_answer="Yale",
        n_agents=3, n_rounds=1, setup=(1, 2),
        correct_pool=CORRECT_POOL, incorrect_pool=INCORRECT_POOL, seed=0,
    )
    assert len(llm.calls) == 0
    assert all(len(agent_hist) == 1 for agent_hist in result.agent_responses)


def test_call_count_scales_with_rounds_not_agents():
    # 5 agents, 3 rounds -> rounds 2 and 3 are each ONE batched call
    # (batching is exactly the point: call count no longer scales with
    # agent count, only with round count) => 2 total calls, not 10.
    llm = FakeLLMClient(response_fn=lambda p: "resp")
    result = run_debate(
        llm=llm, question_id="q1", question="Q", gold_answer="Yale",
        n_agents=5, n_rounds=3, setup=(2, 3),
        correct_pool=CORRECT_POOL, incorrect_pool=INCORRECT_POOL, seed=0,
    )
    assert len(llm.calls) == 2
    assert all(call["method"] == "generate_batch" for call in llm.calls)
    assert all(call["n"] == 5 for call in llm.calls)  # 5 prompts batched per round
    assert len(result.agent_responses) == 5


# ---------------------------------------------------------------------------
# run_debate — fully-connected partner visibility (the actual topology check)
# ---------------------------------------------------------------------------

def test_round2_prompt_excludes_own_response_includes_others():
    captured_prompts = []

    def response_fn(prompt):
        captured_prompts.append(prompt)
        return "revised"

    llm = FakeLLMClient(response_fn=response_fn)
    run_debate(
        llm=llm, question_id="q1", question="Q", gold_answer="Yale",
        n_agents=3, n_rounds=2, setup=(1, 2),
        correct_pool=CORRECT_POOL, incorrect_pool=INCORRECT_POOL, seed=0,
    )

    # 3 round-2 prompts captured (one per agent)
    assert len(captured_prompts) == 3
    for prompt in captured_prompts:
        # exactly 2 "Agent N's response:" blocks per prompt (the OTHER two agents)
        assert prompt.count("'s response:") == 2


def test_fully_connected_all_agents_see_all_others_text():
    # use distinguishable round-1 seeds via a controlled small pool so we
    # can check the exact texts appear in each other agent's round-2 prompt
    distinguishable_correct = ["UNIQUE_CORRECT_TEXT"]
    distinguishable_incorrect = ["UNIQUE_INCORRECT_TEXT"]

    captured_prompts = []

    def response_fn(prompt):
        captured_prompts.append(prompt)
        return "revised"

    llm = FakeLLMClient(response_fn=response_fn)
    run_debate(
        llm=llm, question_id="q1", question="Q", gold_answer="Yale",
        n_agents=2, n_rounds=2, setup=(1, 1),
        correct_pool=distinguishable_correct,
        incorrect_pool=distinguishable_incorrect, seed=0,
    )

    # with only 2 agents, each agent's round-2 prompt must contain the
    # OTHER agent's exact round-1 text
    assert len(captured_prompts) == 2
    combined = " ".join(captured_prompts)
    assert "UNIQUE_CORRECT_TEXT" in combined
    assert "UNIQUE_INCORRECT_TEXT" in combined


def test_custom_communication_fn_overrides_fully_connected():
    # sanity check the hook Module 5 (DIGRA) will use: a communication_fn
    # that always returns an empty partner list regardless of agent count
    def isolated_fn(agent_id, round_idx, prev_round_responses):
        return []

    captured_prompts = []

    def response_fn(prompt):
        captured_prompts.append(prompt)
        return "revised"

    llm = FakeLLMClient(response_fn=response_fn)
    run_debate(
        llm=llm, question_id="q1", question="Q", gold_answer="Yale",
        n_agents=3, n_rounds=2, setup=(1, 2),
        correct_pool=CORRECT_POOL, incorrect_pool=INCORRECT_POOL, seed=0,
        communication_fn=isolated_fn,
    )
    for prompt in captured_prompts:
        assert "no other agents to consider" in prompt


# ---------------------------------------------------------------------------
# DebateResult basic shape
# ---------------------------------------------------------------------------

def test_debate_result_carries_gold_answer_and_alternatives():
    llm = FakeLLMClient(scripted_responses=["r2_a0", "r2_a1", "r2_a2"])
    result = run_debate(
        llm=llm, question_id="q1", question="Q", gold_answer="Yale",
        gold_answer_alternatives=["Yale University"],
        n_agents=3, n_rounds=2, setup=(1, 2),
        correct_pool=CORRECT_POOL, incorrect_pool=INCORRECT_POOL, seed=0,
    )
    assert isinstance(result, DebateResult)
    assert result.gold_answer == "Yale"
    assert result.gold_answer_alternatives == ["Yale University"]


def test_max_tokens_propagates_to_all_debate_llm_calls():
    # regression test for a real bug: run_debate accepted a generation
    # config but never forwarded it, silently defaulting to 1024 regardless
    # of what configs/base.yaml specified.
    llm = FakeLLMClient(response_fn=lambda p: "resp")
    run_debate(
        llm=llm, question_id="q1", question="Q", gold_answer="Yale",
        n_agents=3, n_rounds=3, setup="standard", seed=0,
        max_tokens=42, temperature=0.7, top_p=0.9, top_k=10,
    )
    assert len(llm.calls) > 0
    assert all(call["max_tokens"] == 42 for call in llm.calls)
    assert all(call["temperature"] == 0.7 for call in llm.calls)
