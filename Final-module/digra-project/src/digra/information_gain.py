"""
Information Gain (Eq. 7) and Information Gain Ratio (Eq. 8).

IG_i,t(J)  = H(R_i,t) - H(R_i,t | responses of J)
IGR_i,t(J) = (alpha + IG_i,t(J)) / mean(H(R_j,t) for j in J)

Both are pure arithmetic over already-computed entropy values — the
expensive part (getting H(R_i,t) and H(R_i,t|J) in the first place) lives
in src/digra/partner_selection.py, which calls src/entropy/entropy.py and
the LLMClient. Keeping the formulas here as pure functions makes them
trivial to verify against the paper's own worked numeric example
(Fig. 1(c)), independent of any model or GPU.
"""

from __future__ import annotations


def compute_ig(h_unconditioned: float, h_conditioned: float) -> float:
    """Eq. 7: entropy reduction after conditioning on other agents' responses."""
    return h_unconditioned - h_conditioned


def compute_igr(ig: float, mean_h_j: float, alpha: float) -> float:
    """
    Eq. 8. `mean_h_j` is the mean entropy of the responses in J (the
    candidate partner set), NOT of agent i itself.

    Guards against mean_h_j == 0 (a candidate set whose responses were all
    perfectly certain — entropy exactly 0) with a small epsilon, since the
    literal formula would divide by zero. This is a numerical-safety
    addition beyond the paper's stated formula, not a paper deviation in
    substance — perfectly-zero entropy is a measure-zero edge case in
    practice with top-k logprobs from a real model.
    """
    epsilon = 1e-9
    denom = mean_h_j if mean_h_j > epsilon else epsilon
    return (alpha + ig) / denom
