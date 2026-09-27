import pytest

from src.baselines.mad_variants import (
    build_communication_fn,
    build_random_communication_fn,
    build_sparse_communication_fn,
    run_mad_variant_debate,
)
from src.llm.fake_client import FakeLLMClient


# ---------------------------------------------------------------------------
# build_sparse_communication_fn
# ---------------------------------------------------------------------------

def test_sparse_degree_half_gives_one_partner_for_three_agents():
    # n_agents=3 -> Na-1=2, degree=0.5 -> d=round(1.0)=1 partner each.
    fn = build_sparse_communication_fn(n_agents=3, degree=0.5, seed=0)
    prev = ["resp0", "resp1", "resp2"]
    for agent_id in range(3):
        partners = fn(agent_id, round_idx=1, prev_round_responses=prev)
        assert len(partners) == 1


def test_sparse_degree_zero_gives_no_partners():
    fn = build_sparse_communication_fn(n_agents=4, degree=0.0, seed=0)
    prev = ["a", "b", "c", "d"]
    for agent_id in range(4):
        assert fn(agent_id, round_idx=1, prev_round_responses=prev) == []


def test_sparse_degree_one_is_fully_connected():
    fn = build_sparse_communication_fn(n_agents=4, degree=1.0, seed=0)
    prev = ["a", "b", "c", "d"]
    for agent_id in range(4):
        partners = fn(agent_id, round_idx=1, prev_round_responses=prev)
        assert len(partners) == 3
        assert prev[agent_id] not in partners  # never includes self


def test_sparse_topology_is_fixed_across_rounds():
    fn = build_sparse_communication_fn(n_agents=5, degree=0.5, seed=42)
    prev = ["a", "b", "c", "d", "e"]
    # Sparse topology depends only on agent_id, not round_idx — the same
    # prev-round list must map to the same partner selection at any round.
    assert fn(0, round_idx=1, prev_round_responses=prev) == fn(0, round_idx=2, prev_round_responses=prev)
    assert fn(3, round_idx=1, prev_round_responses=prev) == fn(3, round_idx=5, prev_round_responses=prev)


def test_sparse_is_deterministic_given_same_seed():
    prev = ["a", "b", "c", "d", "e"]
    fn1 = build_sparse_communication_fn(n_agents=5, degree=0.5, seed=7)
    fn2 = build_sparse_communication_fn(n_agents=5, degree=0.5, seed=7)
    assert fn1(2, 1, prev) == fn2(2, 1, prev)


def test_sparse_rejects_out_of_range_degree():
    with pytest.raises(ValueError):
        build_sparse_communication_fn(n_agents=3, degree=1.5, seed=0)
    with pytest.raises(ValueError):
        build_sparse_communication_fn(n_agents=3, degree=-0.1, seed=0)


# ---------------------------------------------------------------------------
# build_random_communication_fn
# ---------------------------------------------------------------------------

def test_random_communication_never_includes_self():
    fn = build_random_communication_fn(n_agents=5, seed=0)
    prev = ["a", "b", "c", "d", "e"]
    for agent_id in range(5):
        for round_idx in range(3):
            partners = fn(agent_id, round_idx, prev)
            assert prev[agent_id] not in partners


def test_random_communication_partner_count_varies_across_rounds():
    fn = build_random_communication_fn(n_agents=6, seed=0)
    prev = ["a", "b", "c", "d", "e", "f"]
    counts = {len(fn(0, r, prev)) for r in range(20)}
    # With 20 independent draws over {0,...,5} possible counts, we should
    # see more than one distinct count (this would only flake with
    # astronomically bad luck given the fixed seed makes it deterministic
    # anyway — same seed always produces the same set of counts).
    assert len(counts) > 1


def test_random_communication_deterministic_given_same_seed():
    prev = ["a", "b", "c", "d"]
    fn1 = build_random_communication_fn(n_agents=4, seed=3)
    fn2 = build_random_communication_fn(n_agents=4, seed=3)
    for round_idx in range(5):
        assert fn1(1, round_idx, prev) == fn2(1, round_idx, prev)


def test_random_communication_differs_across_agents_and_rounds():
    # Different (agent_id, round_idx) pairs must use independent RNG
    # streams, not literally the same draw repeated.
    fn = build_random_communication_fn(n_agents=6, seed=0)
    prev = ["a", "b", "c", "d", "e", "f"]
    draws = {(agent, r): tuple(fn(agent, r, prev)) for agent in range(6) for r in range(4)}
    assert len(set(draws.values())) > 1


# ---------------------------------------------------------------------------
# build_communication_fn dispatch
# ---------------------------------------------------------------------------

def test_dispatch_sparse_requires_sparse_degree():
    with pytest.raises(ValueError, match="sparse_degree"):
        build_communication_fn("mad_sparse_half", n_agents=3, seed=0)


def test_dispatch_random_does_not_require_sparse_degree():
    fn = build_communication_fn("mad_random", n_agents=3, seed=0)
    assert callable(fn)


def test_dispatch_rejects_unknown_variant():
    with pytest.raises(ValueError, match="Unknown MAD variant"):
        build_communication_fn("mad_fully_connected", n_agents=3, seed=0)


# ---------------------------------------------------------------------------
# run_mad_variant_debate (integration through src.agents.debate.run_debate)
# ---------------------------------------------------------------------------

def test_run_mad_variant_debate_sparse_produces_standard_debate_result():
    llm = FakeLLMClient(scripted_responses=["Final answer: Yale"] * 100)
    result = run_mad_variant_debate(
        llm, "mad_sparse_half", "q1", "Who won?", "Yale",
        n_agents=3, n_rounds=3, sparse_degree=0.5, seed=0,
    )
    assert result.setup == "standard"
    assert result.n_agents == 3
    assert result.n_rounds == 3
    assert len(result.agent_responses) == 3
    assert all(len(rounds) == 3 for rounds in result.agent_responses)


def test_run_mad_variant_debate_random_produces_standard_debate_result():
    llm = FakeLLMClient(scripted_responses=["Final answer: Yale"] * 100)
    result = run_mad_variant_debate(
        llm, "mad_random", "q1", "Who won?", "Yale",
        n_agents=3, n_rounds=3, seed=0,
    )
    assert result.setup == "standard"
    assert result.n_agents == 3


def test_run_mad_variant_debate_rejects_unknown_variant():
    llm = FakeLLMClient(scripted_responses=["Final answer: Yale"] * 100)
    with pytest.raises(ValueError):
        run_mad_variant_debate(
            llm, "mad_fully_connected", "q1", "Who won?", "Yale",
            n_agents=3, n_rounds=3, seed=0,
        )
