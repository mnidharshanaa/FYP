import random

import pytest

from src.agents.round_seeding import InsufficientPoolError, seed_round_one_from_pool

CORRECT_POOL = ["correct 1", "correct 2", "correct 3", "correct 4", "correct 5"]
INCORRECT_POOL = ["appeal 1", "appeal 2", "appeal 3"]


def test_seed_round_one_returns_correct_total_count():
    rng = random.Random(0)
    result = seed_round_one_from_pool(2, 1, CORRECT_POOL, INCORRECT_POOL, rng)
    assert len(result) == 3


def test_seed_round_one_upper_bound_all_correct():
    rng = random.Random(0)
    result = seed_round_one_from_pool(0, 3, CORRECT_POOL, INCORRECT_POOL, rng)
    assert len(result) == 3
    assert all(r in CORRECT_POOL for r in result)


def test_seed_round_one_lower_bound_all_incorrect():
    rng = random.Random(0)
    result = seed_round_one_from_pool(3, 0, CORRECT_POOL, INCORRECT_POOL, rng)
    assert len(result) == 3
    assert all(r in INCORRECT_POOL for r in result)


def test_seed_round_one_deterministic_given_seed():
    result_a = seed_round_one_from_pool(2, 1, CORRECT_POOL, INCORRECT_POOL, random.Random(42))
    result_b = seed_round_one_from_pool(2, 1, CORRECT_POOL, INCORRECT_POOL, random.Random(42))
    assert result_a == result_b


def test_seed_round_one_incorrect_pool_empty_raises():
    rng = random.Random(0)
    with pytest.raises(InsufficientPoolError):
        seed_round_one_from_pool(1, 0, CORRECT_POOL, [], rng)


def test_seed_round_one_correct_pool_empty_raises():
    rng = random.Random(0)
    with pytest.raises(InsufficientPoolError):
        seed_round_one_from_pool(0, 1, [], INCORRECT_POOL, rng)


def test_seed_round_one_zero_zero_needs_no_pools():
    # degenerate but shouldn't crash even with empty pools, since nothing is drawn
    rng = random.Random(0)
    result = seed_round_one_from_pool(0, 0, [], [], rng)
    assert result == []


def test_seed_round_one_5_incorrect_from_3_appeals_samples_with_replacement():
    # 5-agent lower-bound setup, only 3 distinct FARM appeals available
    rng = random.Random(0)
    result = seed_round_one_from_pool(5, 0, CORRECT_POOL, INCORRECT_POOL, rng)
    assert len(result) == 5
    assert all(r in INCORRECT_POOL for r in result)
    # with only 3 distinct options and 5 draws, at least one must repeat
    assert len(set(result)) < 5


def test_seed_round_one_output_is_shuffled_not_incorrect_then_correct_always():
    # run many seeds and confirm the incorrect-first ordering isn't fixed
    orders_with_incorrect_first = 0
    trials = 30
    for seed in range(trials):
        rng = random.Random(seed)
        result = seed_round_one_from_pool(1, 1, CORRECT_POOL, INCORRECT_POOL, rng)
        if result[0] in INCORRECT_POOL:
            orders_with_incorrect_first += 1
    # should not be deterministically always-first or always-last across seeds
    assert 0 < orders_with_incorrect_first < trials
