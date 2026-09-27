import pytest

from src.reliability.information_flow import (
    classify_information_flow,
    compute_bhp,
    compute_flow_rates,
    compute_flow_rates_by_round,
    compute_hpr,
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


# ---------------------------------------------------------------------------
# compute_hpr / compute_bhp
# ---------------------------------------------------------------------------

def _comm(origin_correct, target_correct_before, result_correct, decision):
    return {
        "origin_correct": origin_correct, "target_correct_before": target_correct_before,
        "result_correct": result_correct, "decision": decision,
    }


def test_hpr_only_counts_wrong_source_correct_target_opportunities():
    comms = [
        _comm(origin_correct=False, target_correct_before=True, result_correct=False, decision="propagate"),  # opportunity, harm realized
        _comm(origin_correct=False, target_correct_before=True, result_correct=True, decision="suppress"),   # opportunity, harm avoided
        _comm(origin_correct=True, target_correct_before=True, result_correct=False, decision="propagate"),  # NOT an opportunity (origin correct)
        _comm(origin_correct=False, target_correct_before=False, result_correct=False, decision="propagate"),  # NOT an opportunity (target already wrong)
    ]
    assert compute_hpr(comms) == pytest.approx(0.5)  # 1 of 2 true opportunities resulted in harm


def test_hpr_none_when_no_opportunities():
    comms = [_comm(origin_correct=True, target_correct_before=True, result_correct=True, decision="propagate")]
    assert compute_hpr(comms) is None


def test_hpr_ignores_opportunities_with_unknown_result():
    comms = [_comm(origin_correct=False, target_correct_before=True, result_correct=None, decision="propagate")]
    assert compute_hpr(comms) is None


def test_bhp_measures_blocking_rate_not_outcome():
    comms = [
        _comm(origin_correct=False, target_correct_before=True, result_correct=False, decision="propagate"),  # opportunity, NOT blocked
        _comm(origin_correct=False, target_correct_before=True, result_correct=None, decision="suppress"),    # opportunity, blocked (outcome unknown, still counts)
        _comm(origin_correct=False, target_correct_before=True, result_correct=None, decision="verify"),      # opportunity, NOT blocked (verify != propagate... wait)
    ]
    # "verify" is not "propagate" either, so it counts as blocked here too —
    # BHP is about whether the gate let it through unconditionally.
    assert compute_bhp(comms) == pytest.approx(2 / 3)


def test_bhp_none_when_no_opportunities():
    comms = [_comm(origin_correct=True, target_correct_before=True, result_correct=True, decision="propagate")]
    assert compute_bhp(comms) is None


def test_bhp_does_not_require_known_result_correct():
    # BHP is about the DECISION, not the aftermath — unlike HPR, an
    # unresolved/unknown result_correct must not exclude it from BHP.
    comms = [_comm(origin_correct=False, target_correct_before=True, result_correct=None, decision="suppress")]
    assert compute_bhp(comms) == pytest.approx(1.0)
