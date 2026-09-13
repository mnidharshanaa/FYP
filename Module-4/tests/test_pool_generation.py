from src.data.pool_generation import build_pool_for_question
from src.llm.fake_client import FakeLLMClient


def test_sufficient_samples_skips_guided_batch():
    # n_attempts=10, n_all=3, 5 correct + 5 wrong in the single generate() call
    scripted = ["Final answer: Yale"] * 5 + ["Final answer: Duke"] * 5
    llm = FakeLLMClient(scripted_responses=scripted)

    result = build_pool_for_question(
        llm=llm,
        question_id="q1",
        question="Who won?",
        gold_answer="Yale",
        alternatives=[],
        logical_appeals=["appeal 1", "appeal 2"],
        n_all=3,
        n_attempts=10,
        seed=0,
    )

    assert result.n_correct_sampled == 5
    assert result.difficulty == 0.5
    assert len(result.correct_texts) == 3
    assert result.n_generated_via_fewshot == 0
    assert result.fully_satisfied
    # only ONE generate() call — the guided batch must not have fired
    assert len(llm.calls) == 1
    assert result.incorrect_texts == ["appeal 1", "appeal 2"]


def test_insufficient_samples_triggers_guided_batch():
    # n_attempts=10, n_all=5, only 2 correct -> need 3 more via guided batch
    scripted_step1 = ["Final answer: Yale"] * 2 + ["Final answer: Duke"] * 8
    scripted_step3 = ["Final answer: Yale (guided)"] * 3
    llm = FakeLLMClient(scripted_responses=scripted_step1 + scripted_step3)

    result = build_pool_for_question(
        llm=llm,
        question_id="q2",
        question="Who won?",
        gold_answer="Yale",
        alternatives=[],
        logical_appeals=[],
        n_all=5,
        n_attempts=10,
        seed=0,
    )

    assert result.n_correct_sampled == 2
    assert result.n_generated_via_fewshot == 3
    assert len(result.correct_texts) == 5
    assert result.fully_satisfied
    # two generate() calls: initial 10-sample draw + guided batch of 3
    assert len(llm.calls) == 2
    assert llm.calls[0]["n"] == 10
    assert llm.calls[1]["n"] == 3


def test_guided_batch_prompt_contains_gold_answer_and_examples():
    scripted_step1 = ["Final answer: Duke"] * 10  # 0 correct -> need all 5 via guided batch
    scripted_step3 = ["Final answer: Yale"] * 5
    llm = FakeLLMClient(scripted_responses=scripted_step1 + scripted_step3)

    build_pool_for_question(
        llm=llm, question_id="q3", question="Who won the championship?",
        gold_answer="Yale", alternatives=[], logical_appeals=[],
        n_all=5, n_attempts=10, seed=0,
    )

    guided_prompt = llm.calls[1]["prompt"]
    assert "Who won the championship?" in guided_prompt
    assert "Yale" in guided_prompt


def test_zero_correct_in_step1_still_reaches_n_all_via_guided_batch():
    scripted_step1 = ["Final answer: Duke"] * 10
    scripted_step3 = ["Final answer: Yale"] * 5
    llm = FakeLLMClient(scripted_responses=scripted_step1 + scripted_step3)

    result = build_pool_for_question(
        llm=llm, question_id="q4", question="Q", gold_answer="Yale",
        alternatives=[], logical_appeals=[], n_all=5, n_attempts=10, seed=0,
    )

    assert result.n_correct_sampled == 0
    assert result.n_generated_via_fewshot == 5
    assert len(result.correct_texts) == 5
    assert result.fully_satisfied


def test_incorrect_texts_passed_through_unchanged():
    scripted = ["Final answer: Yale"] * 10
    llm = FakeLLMClient(scripted_responses=scripted)

    result = build_pool_for_question(
        llm=llm, question_id="q5", question="Q", gold_answer="Yale",
        alternatives=[], logical_appeals=["a1", "a2", "a3"],
        n_all=3, n_attempts=10, seed=0,
    )
    assert result.incorrect_texts == ["a1", "a2", "a3"]


def test_max_tokens_actually_propagates_to_llm_calls():
    # regression test for a real bug: max_tokens was accepted by config but
    # never forwarded to llm.generate(), silently falling back to
    # LLMClient's default (1024) regardless of what was configured.
    scripted = ["Final answer: Yale"] * 5 + ["Final answer: Duke"] * 5
    llm = FakeLLMClient(scripted_responses=scripted)

    build_pool_for_question(
        llm=llm, question_id="q1", question="Q", gold_answer="Yale",
        alternatives=[], logical_appeals=[], n_all=3, n_attempts=10,
        seed=0, max_tokens=42, temperature=0.7, top_p=0.9, top_k=10,
    )

    assert llm.calls[0]["max_tokens"] == 42
    assert llm.calls[0]["temperature"] == 0.7
    assert llm.calls[0]["top_p"] == 0.9
    assert llm.calls[0]["top_k"] == 10


def test_max_tokens_propagates_to_guided_batch_too():
    scripted_step1 = ["Final answer: Duke"] * 10
    scripted_step3 = ["Final answer: Yale"] * 5
    llm = FakeLLMClient(scripted_responses=scripted_step1 + scripted_step3)

    build_pool_for_question(
        llm=llm, question_id="q1", question="Q", gold_answer="Yale",
        alternatives=[], logical_appeals=[], n_all=5, n_attempts=10,
        seed=0, max_tokens=42,
    )
    assert llm.calls[1]["max_tokens"] == 42


def test_result_is_deterministic_given_same_seed():
    scripted = ["Final answer: Yale"] * 5 + ["Final answer: Duke"] * 5
    llm_a = FakeLLMClient(scripted_responses=list(scripted))
    llm_b = FakeLLMClient(scripted_responses=list(scripted))

    result_a = build_pool_for_question(
        llm=llm_a, question_id="q6", question="Q", gold_answer="Yale",
        alternatives=[], logical_appeals=[], n_all=3, n_attempts=10, seed=42,
    )
    result_b = build_pool_for_question(
        llm=llm_b, question_id="q6", question="Q", gold_answer="Yale",
        alternatives=[], logical_appeals=[], n_all=3, n_attempts=10, seed=42,
    )
    assert result_a.correct_texts == result_b.correct_texts
