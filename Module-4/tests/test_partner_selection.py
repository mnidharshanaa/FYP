import pytest

from src.digra.partner_selection import enumerate_candidate_subsets, select_best_partners


def test_enumerate_subsets_all_nonempty():
    subsets = list(enumerate_candidate_subsets([1, 2]))
    assert frozenset({1}) in subsets
    assert frozenset({2}) in subsets
    assert frozenset({1, 2}) in subsets
    assert len(subsets) == 3  # 2^2 - 1


def test_enumerate_subsets_three_candidates():
    subsets = list(enumerate_candidate_subsets([1, 2, 3]))
    assert len(subsets) == 7  # 2^3 - 1


def test_enumerate_subsets_respects_max_size_cap():
    subsets = list(enumerate_candidate_subsets([1, 2, 3], max_subset_size=1))
    assert len(subsets) == 3  # only singletons
    assert all(len(s) == 1 for s in subsets)


def test_enumerate_subsets_cap_larger_than_n_is_noop():
    subsets = list(enumerate_candidate_subsets([1, 2], max_subset_size=100))
    assert len(subsets) == 3


def test_select_best_partners_picks_highest_igr():
    # candidate 1 alone: IG=0.5, H=0.5 -> IGR=(0.2+0.5)/0.5=1.4
    # candidate 2 alone: IG=0.1, H=0.5 -> IGR=(0.2+0.1)/0.5=0.6
    # -> subset {1} should win
    def ig_fn(subset):
        return {frozenset({1}): 0.5, frozenset({2}): 0.1, frozenset({1, 2}): 0.3}[subset]

    entropy_by_agent = {1: 0.5, 2: 0.5}
    result = select_best_partners(
        candidate_ids=[1, 2], entropy_by_agent=entropy_by_agent,
        ig_fn=ig_fn, alpha=0.2,
    )
    assert result.best_subset == frozenset({1})


def test_select_best_partners_penalizes_high_entropy_candidate():
    # both candidates give identical IG, but candidate 2 has much higher
    # entropy (more likely hallucinating) -> IGR should favor candidate 1
    def ig_fn(subset):
        return 0.2  # same IG regardless of subset

    entropy_by_agent = {1: 0.3, 2: 0.9}
    result = select_best_partners(
        candidate_ids=[1, 2], entropy_by_agent=entropy_by_agent,
        ig_fn=ig_fn, alpha=0.2,
    )
    assert result.best_subset == frozenset({1})


def test_select_best_partners_use_ratio_false_ignores_entropy_penalty():
    # Same setup as test_select_best_partners_penalizes_high_entropy_candidate
    # (identical IG, candidate 2 has much higher entropy) but with
    # use_ratio=False (the DIG ablation): raw IG is used directly, so the
    # entropy difference must NOT change the outcome — this is the
    # precise, single behavioral difference between DIG and DIGRA.
    def ig_fn(subset):
        return {frozenset({1}): 0.2, frozenset({2}): 0.5, frozenset({1, 2}): 0.3}[subset]

    entropy_by_agent = {1: 0.3, 2: 0.9}
    result = select_best_partners(
        candidate_ids=[1, 2], entropy_by_agent=entropy_by_agent,
        ig_fn=ig_fn, alpha=0.2, use_ratio=False,
    )
    # candidate 2 has the higher raw IG (0.5 > 0.2) and should win despite
    # its much higher entropy, since use_ratio=False never looks at entropy
    assert result.best_subset == frozenset({2})
    assert result.best_igr == 0.5  # raw IG, not run through compute_igr at all


def test_select_best_partners_use_ratio_true_is_the_default():
    # use_ratio's default (True) must reproduce prior behavior exactly —
    # this guards against the DIG addition silently changing DIGRA itself.
    def ig_fn(subset):
        return {frozenset({1}): 0.5, frozenset({2}): 0.1, frozenset({1, 2}): 0.3}[subset]

    entropy_by_agent = {1: 0.5, 2: 0.5}
    result = select_best_partners(
        candidate_ids=[1, 2], entropy_by_agent=entropy_by_agent, ig_fn=ig_fn, alpha=0.2,
    )
    assert result.best_subset == frozenset({1})
    assert abs(result.best_igr - 1.4) < 1e-9  # (0.2 + 0.5) / 0.5


def test_select_best_partners_records_all_scores():
    def ig_fn(subset):
        return 0.1

    entropy_by_agent = {1: 0.5, 2: 0.5}
    result = select_best_partners(
        candidate_ids=[1, 2], entropy_by_agent=entropy_by_agent,
        ig_fn=ig_fn, alpha=0.2,
    )
    assert len(result.all_scores) == 3  # {1}, {2}, {1,2}


def test_select_best_partners_empty_candidates_raises():
    with pytest.raises(ValueError):
        select_best_partners(
            candidate_ids=[], entropy_by_agent={}, ig_fn=lambda s: 0.0, alpha=0.2,
        )


def test_select_best_partners_single_candidate_trivial():
    def ig_fn(subset):
        return 0.3

    result = select_best_partners(
        candidate_ids=[5], entropy_by_agent={5: 0.4}, ig_fn=ig_fn, alpha=0.2,
    )
    assert result.best_subset == frozenset({5})


def test_select_best_partners_respects_max_subset_size():
    calls = []

    def ig_fn(subset):
        calls.append(subset)
        return len(subset) * 0.1  # larger subsets "look" better by IG alone

    entropy_by_agent = {1: 0.5, 2: 0.5, 3: 0.5}
    result = select_best_partners(
        candidate_ids=[1, 2, 3], entropy_by_agent=entropy_by_agent,
        ig_fn=ig_fn, alpha=0.2, max_subset_size=1,
    )
    # only singleton subsets should have been evaluated
    assert all(len(s) == 1 for s in calls)
    assert len(result.best_subset) == 1
