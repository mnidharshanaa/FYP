import math

import pytest

from src.reliability.propagation_score import (
    classify_decision,
    classify_quadrant,
    normalize_utility,
    propagation_score,
    score_and_decide,
)


# ---------------------------------------------------------------------------
# normalize_utility
# ---------------------------------------------------------------------------

def test_normalize_utility_min_maps_to_zero():
    assert normalize_utility(igr=1.0, igr_min=1.0, igr_max=5.0) == pytest.approx(0.0, abs=1e-6)


def test_normalize_utility_max_maps_to_near_one():
    # epsilon in the denominator means igr_max maps to slightly under 1.0,
    # not exactly 1.0 — this is intentional (see docstring), just confirm
    # it's very close rather than asserting exact equality.
    u = normalize_utility(igr=5.0, igr_min=1.0, igr_max=5.0)
    assert u == pytest.approx(1.0, abs=1e-6)


def test_normalize_utility_midpoint():
    u = normalize_utility(igr=3.0, igr_min=1.0, igr_max=5.0)
    assert u == pytest.approx(0.5, abs=1e-3)


def test_normalize_utility_clips_out_of_pool_values():
    # An igr below igr_min or above igr_max (e.g. a value from a
    # different comparison pool) should clip into [0, 1], not go negative
    # or exceed 1.
    assert normalize_utility(igr=-10, igr_min=0, igr_max=1) == 0.0
    assert normalize_utility(igr=10, igr_min=0, igr_max=1) == 1.0


def test_normalize_utility_rejects_inverted_bounds():
    with pytest.raises(ValueError):
        normalize_utility(igr=1.0, igr_min=5.0, igr_max=1.0)


# ---------------------------------------------------------------------------
# propagation_score
# ---------------------------------------------------------------------------

def test_propagation_score_matches_spec_worked_example():
    assert propagation_score(0.90, 0.90, 0.5) == pytest.approx(0.900, abs=1e-3)
    assert propagation_score(0.90, 0.20, 0.5) == pytest.approx(0.424, abs=1e-3)
    assert propagation_score(0.20, 0.90, 0.5) == pytest.approx(0.424, abs=1e-3)
    assert propagation_score(0.20, 0.20, 0.5) == pytest.approx(0.200, abs=1e-3)


def test_propagation_score_lam_zero_is_reliability_only():
    assert propagation_score(u=0.0, r=0.7, lam=0.0) == pytest.approx(0.7)
    assert propagation_score(u=1.0, r=0.7, lam=0.0) == pytest.approx(0.7)


def test_propagation_score_lam_one_is_utility_only():
    assert propagation_score(u=0.7, r=0.0, lam=1.0) == pytest.approx(0.7)
    assert propagation_score(u=0.7, r=1.0, lam=1.0) == pytest.approx(0.7)


def test_propagation_score_geometric_form_penalizes_one_weak_signal_harder_than_arithmetic_mean():
    ps = propagation_score(0.9, 0.2, 0.5)
    arithmetic_mean = (0.9 + 0.2) / 2
    assert ps < arithmetic_mean


def test_propagation_score_rejects_out_of_range_inputs():
    with pytest.raises(ValueError):
        propagation_score(1.5, 0.5, 0.5)
    with pytest.raises(ValueError):
        propagation_score(0.5, -0.1, 0.5)
    with pytest.raises(ValueError):
        propagation_score(0.5, 0.5, 1.1)


# ---------------------------------------------------------------------------
# classify_decision
# ---------------------------------------------------------------------------

def test_classify_decision_propagate_when_ps_above_threshold():
    assert classify_decision(ps=0.8, u=0.9, tau_p=0.6, tau_u=0.5) == "propagate"


def test_classify_decision_verify_when_ps_low_but_utility_high():
    assert classify_decision(ps=0.3, u=0.9, tau_p=0.6, tau_u=0.5) == "verify"


def test_classify_decision_suppress_when_both_low():
    assert classify_decision(ps=0.3, u=0.2, tau_p=0.6, tau_u=0.5) == "suppress"


def test_classify_decision_boundary_is_inclusive_propagate():
    assert classify_decision(ps=0.6, u=0.0, tau_p=0.6, tau_u=0.5) == "propagate"


def test_classify_decision_boundary_is_inclusive_verify():
    assert classify_decision(ps=0.59, u=0.5, tau_p=0.6, tau_u=0.5) == "verify"


# ---------------------------------------------------------------------------
# classify_quadrant
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("u,r,expected", [
    (0.9, 0.9, "propagate"),
    (0.9, 0.1, "verify"),
    (0.1, 0.9, "low_priority"),
    (0.1, 0.1, "suppress"),
])
def test_classify_quadrant_all_four_cells(u, r, expected):
    assert classify_quadrant(u, r) == expected


def test_classify_quadrant_custom_thresholds():
    # With a high utility threshold, a middling U=0.6 counts as "low".
    assert classify_quadrant(u=0.6, r=0.9, u_threshold=0.8, r_threshold=0.5) == "low_priority"


def test_classify_quadrant_is_independent_of_propagation_score():
    # Sanity: quadrant never takes ps/tau_p/tau_u — verifies the two
    # classifiers really are decoupled, not accidentally coupled through
    # shared defaults.
    import inspect
    sig = inspect.signature(classify_quadrant)
    assert "ps" not in sig.parameters
    assert "tau_p" not in sig.parameters


# ---------------------------------------------------------------------------
# score_and_decide (convenience wrapper)
# ---------------------------------------------------------------------------

def test_score_and_decide_bundles_consistent_results():
    record = score_and_decide(u=0.9, r=0.9, lam=0.5, tau_p=0.6, tau_u=0.5)
    assert record.ps == pytest.approx(propagation_score(0.9, 0.9, 0.5))
    assert record.decision == classify_decision(record.ps, 0.9, 0.6, 0.5)
    assert record.quadrant == classify_quadrant(0.9, 0.9)


def test_score_and_decide_high_utility_low_reliability_is_verify_case():
    # This is the spec's flagship "why we need conditional RAG" case.
    record = score_and_decide(u=0.91, r=0.28, lam=0.5, tau_p=0.6, tau_u=0.5)
    assert record.decision == "verify"
    assert record.quadrant == "verify"
    assert record.ps == pytest.approx(math.sqrt(0.91 * 0.28), abs=1e-6)
