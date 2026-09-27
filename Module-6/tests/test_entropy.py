import math

import pytest

from src.entropy.entropy import mean_token_entropy


def test_none_input_returns_zero():
    assert mean_token_entropy(None) == 0.0


def test_empty_list_returns_zero():
    assert mean_token_entropy([]) == 0.0


def test_all_empty_positions_returns_zero():
    assert mean_token_entropy([{}, {}]) == 0.0


def test_single_certain_token_has_near_zero_entropy():
    # logprob 0.0 == probability 1.0 -> perfectly certain -> H = 0
    result = mean_token_entropy([{"the": 0.0}])
    assert result == pytest.approx(0.0, abs=1e-9)


def test_two_equal_probability_tokens_matches_hand_computed_entropy():
    # p=0.5 each -> H = -2 * 0.5*ln(0.5) = ln(2)
    logprob_half = math.log(0.5)
    result = mean_token_entropy([{"a": logprob_half, "b": logprob_half}])
    assert result == pytest.approx(math.log(2), abs=1e-9)


def test_averages_across_multiple_positions():
    logprob_half = math.log(0.5)
    certain = {"x": 0.0}
    uncertain = {"a": logprob_half, "b": logprob_half}
    result = mean_token_entropy([certain, uncertain])
    expected = (0.0 + math.log(2)) / 2
    assert result == pytest.approx(expected, abs=1e-9)


def test_empty_positions_are_skipped_not_counted_as_zero():
    logprob_half = math.log(0.5)
    uncertain = {"a": logprob_half, "b": logprob_half}
    # one empty position mixed in — should NOT drag the average toward 0
    result = mean_token_entropy([uncertain, {}])
    assert result == pytest.approx(math.log(2), abs=1e-9)


def test_higher_uncertainty_yields_higher_entropy():
    logprob_half = math.log(0.5)
    logprob_quarter = math.log(0.25)
    two_way = mean_token_entropy([{"a": logprob_half, "b": logprob_half}])
    four_way = mean_token_entropy([
        {"a": logprob_quarter, "b": logprob_quarter, "c": logprob_quarter, "d": logprob_quarter}
    ])
    assert four_way > two_way
