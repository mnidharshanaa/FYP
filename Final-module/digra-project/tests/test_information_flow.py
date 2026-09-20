import pytest

from src.reliability.information_flow import (
    classify_information_flow,
    compute_flow_rates,
    compute_flow_rates_by_round,
)


# ---------------------------------------------------------------------------
# classify_information_flow
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("origin,result,expected", [
    (True, True, "C->C"),
    (True, False, "C->W"),
    (False, True, "W->C"),
    (False, False, "W->W"),
])
def test_classify_information_flow_all_four_cases(origin, result, expected):
    assert classify_information_flow(origin, result) == expected


# ---------------------------------------------------------------------------
# compute_flow_rates
# ---------------------------------------------------------------------------

def test_compute_flow_rates_basic_counts_and_rates():
    flows = ["W->W", "W->W", "W->C", "C->C", "C->W"]
    result = compute_flow_rates(flows)
    assert result["counts"] == {"C->C": 1, "C->W": 1, "W->C": 1, "W->W": 2}
    # wrong_origin = 2 (W->W) + 1 (W->C) = 3
    assert result["WWR"] == pytest.approx(2 / 3)
    assert result["WCR"] == pytest.approx(1 / 3)
    # correct_origin = 1 (C->C) + 1 (C->W) = 2
    assert result["CWR"] == pytest.approx(1 / 2)


def test_compute_flow_rates_none_when_no_wrong_origin_communications():
    flows = ["C->C", "C->C", "C->W"]
    result = compute_flow_rates(flows)
    assert result["WWR"] is None
    assert result["WCR"] is None
    assert result["CWR"] == pytest.approx(1 / 3)


def test_compute_flow_rates_none_when_no_correct_origin_communications():
    flows = ["W->W", "W->C"]
    result = compute_flow_rates(flows)
    assert result["CWR"] is None
    assert result["WWR"] == pytest.approx(0.5)
    assert result["WCR"] == pytest.approx(0.5)


def test_compute_flow_rates_empty_input_returns_all_none():
    result = compute_flow_rates([])
    assert result["WWR"] is None
    assert result["WCR"] is None
    assert result["CWR"] is None
    assert result["counts"] == {"C->C": 0, "C->W": 0, "W->C": 0, "W->W": 0}


def test_compute_flow_rates_rejects_unknown_label():
    with pytest.raises(ValueError):
        compute_flow_rates(["C->C", "bogus"])


# ---------------------------------------------------------------------------
# compute_flow_rates_by_round
# ---------------------------------------------------------------------------

def test_compute_flow_rates_by_round_keys_match_input_rounds():
    flows_by_round = {
        1: ["C->C", "W->W"],
        2: ["W->C", "W->C"],
        3: ["C->W"],
    }
    result = compute_flow_rates_by_round(flows_by_round)
    assert set(result.keys()) == {1, 2, 3}
    assert result[2]["WCR"] == pytest.approx(1.0)
    assert result[3]["CWR"] == pytest.approx(1.0)
