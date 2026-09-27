import random

import pytest

from src.data.pool_builder import difficulty_score, select_correct_pool


def test_difficulty_score_basic():
    assert difficulty_score(25, 50) == 0.5


def test_difficulty_score_zero_attempts_raises():
    with pytest.raises(ValueError):
        difficulty_score(0, 0)


def _texts(n_correct, n_wrong, correct_answer="Yale", wrong_answer="Duke"):
    correct = [f"Reasoning... Final answer: {correct_answer}" for _ in range(n_correct)]
    wrong = [f"Reasoning... Final answer: {wrong_answer}" for _ in range(n_wrong)]
    return correct + wrong


def test_select_correct_pool_enough_samples_retains_exactly_n_all():
    # 30 correct out of 50 (n1=30 >= n_all=5) -> step 2 branch
    texts = _texts(n_correct=30, n_wrong=20)
    rng = random.Random(0)
    result = select_correct_pool(texts, gold_answer="Yale", alternatives=[], n_all=5, rng=rng)

    assert result.n_correct_sampled == 30
    assert result.n_attempts == 50
    assert result.difficulty == 0.6
    assert result.n_needed == 0
    assert len(result.correct_texts) == 5
    assert all("Yale" in t for t in result.correct_texts)


def test_select_correct_pool_insufficient_samples_reports_n_needed():
    # only 2 correct out of 50 (n1=2 < n_all=5) -> step 3 territory
    texts = _texts(n_correct=2, n_wrong=48)
    rng = random.Random(0)
    result = select_correct_pool(texts, gold_answer="Yale", alternatives=[], n_all=5, rng=rng)

    assert result.n_correct_sampled == 2
    assert result.difficulty == 0.04
    assert result.n_needed == 3
    assert len(result.correct_texts) == 2  # keeps everything correct, no truncation


def test_select_correct_pool_exactly_n_all_correct():
    texts = _texts(n_correct=5, n_wrong=45)
    rng = random.Random(0)
    result = select_correct_pool(texts, gold_answer="Yale", alternatives=[], n_all=5, rng=rng)

    assert result.n_needed == 0
    assert len(result.correct_texts) == 5


def test_select_correct_pool_zero_correct():
    texts = _texts(n_correct=0, n_wrong=50)
    rng = random.Random(0)
    result = select_correct_pool(texts, gold_answer="Yale", alternatives=[], n_all=5, rng=rng)

    assert result.n_correct_sampled == 0
    assert result.difficulty == 0.0
    assert result.n_needed == 5
    assert result.correct_texts == []


def test_select_correct_pool_is_deterministic_given_seeded_rng():
    texts = _texts(n_correct=30, n_wrong=20)
    result_a = select_correct_pool(texts, "Yale", [], 5, random.Random(42))
    result_b = select_correct_pool(texts, "Yale", [], 5, random.Random(42))
    assert result_a.correct_texts == result_b.correct_texts


def test_select_correct_pool_uses_alternatives():
    texts = [
        "Final answer: Osama bin Laden caused the 9/11 attacks",
    ] * 10 + ["Final answer: aliens"] * 40
    rng = random.Random(0)
    result = select_correct_pool(
        texts,
        gold_answer="Al-Qaeda caused the 9/11 attacks",
        alternatives=["Osama bin Laden caused the 9/11 attacks"],
        n_all=5,
        rng=rng,
    )
    assert result.n_correct_sampled == 10
