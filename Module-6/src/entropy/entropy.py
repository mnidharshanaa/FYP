"""
Mean token entropy (DIGRA paper Eq. 6).

H(X) = -sum_x p(x) log p(x), averaged across generated token positions.

Important limitation, stated explicitly rather than glossed over: we only
ever have TOP-K logprobs from vLLM (see GenerationResult.token_logprobs),
never the full vocabulary distribution. This is a top-k approximation of
entropy, not exact entropy — probability mass outside the top-k is
invisible to this computation. This is standard practice for this kind of
estimate and is what the DIGRA paper's own Table VIII sensitivity analysis
implicitly relies on too (they note their method tolerates approximate
entropy estimates, since partner selection only needs entropy
*differences* within a comparison set to be roughly right, not exact).
"""

from __future__ import annotations

import math
from typing import Optional


def mean_token_entropy(token_logprobs: Optional[list]) -> float:
    """
    token_logprobs: list of per-position dicts, e.g.
        [{"Yale": -0.1, "Duke": -2.3, ...}, {...}, ...]
    (natural-log logprobs, as returned by vLLM and GenerationResult).

    Positions with an empty dict (e.g. the first position of a forced-decode
    call, which has no preceding-token probability — see
    VLLMClient.forced_decode) are skipped rather than treated as zero
    entropy, since an empty dict doesn't represent a real distribution.

    Returns 0.0 for None/empty input or if every position was skipped —
    documented as the "no information" default rather than raising, since
    callers (IG computation) need a numeric value to do arithmetic with.
    """
    if not token_logprobs:
        return 0.0

    position_entropies = []
    for position in token_logprobs:
        if not position:
            continue
        probs = [math.exp(lp) for lp in position.values()]
        h = -sum(p * math.log(p) for p in probs if p > 0.0)
        position_entropies.append(h)

    if not position_entropies:
        return 0.0
    return sum(position_entropies) / len(position_entropies)
