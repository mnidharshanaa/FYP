import pytest

from src.digra.information_gain import compute_ig, compute_igr


def test_ig_positive_when_conditioning_reduces_entropy():
    # H drops from 0.9 to 0.7 after conditioning -> positive information gain
    assert compute_ig(h_unconditioned=0.9, h_conditioned=0.7) == pytest.approx(0.2)


def test_ig_negative_when_conditioning_increases_entropy():
    # a misleading partner can make the agent LESS certain -> negative IG
    assert compute_ig(h_unconditioned=0.5, h_conditioned=0.8) == pytest.approx(-0.3)


def test_ig_zero_when_no_change():
    assert compute_ig(h_unconditioned=0.5, h_conditioned=0.5) == pytest.approx(0.0)


def test_igr_hand_computed():
    # IGR = (alpha + IG) / mean_h_j
    # (0.2 + 0.2) / 0.5 = 0.8
    assert compute_igr(ig=0.2, mean_h_j=0.5, alpha=0.2) == pytest.approx(0.8)


def test_igr_lower_partner_entropy_yields_higher_igr():
    # same IG, less-uncertain partner set -> higher IGR (rewards confident,
    # informative partners over noisy ones)
    confident_partner_igr = compute_igr(ig=0.15, mean_h_j=0.5, alpha=0.2)
    noisy_partner_igr = compute_igr(ig=0.15, mean_h_j=0.9, alpha=0.2)
    assert confident_partner_igr > noisy_partner_igr


def test_igr_zero_mean_entropy_does_not_divide_by_zero():
    # defensive guard against the literal formula's division by zero
    result = compute_igr(ig=0.2, mean_h_j=0.0, alpha=0.2)
    assert result > 0
    assert result != float("inf")


def test_igr_negative_ig_can_still_be_positive_if_alpha_dominates():
    # alpha=0.2 partially offsets a small negative IG, per the paper's
    # stated purpose for alpha (balance parameter)
    result = compute_igr(ig=-0.1, mean_h_j=0.5, alpha=0.2)
    assert result == pytest.approx((0.2 - 0.1) / 0.5)
