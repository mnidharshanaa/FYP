from src.memory.trust import (
    TrustTracker,
    compute_flag,
    cross_round_inconsistent,
)


# ---------------------------------------------------------------------------
# TrustTracker
# ---------------------------------------------------------------------------

def test_new_agent_starts_at_initial_trust():
    tracker = TrustTracker(initial_trust=1.0)
    assert tracker.get("agent_1") == 1.0


def test_flag_decays_trust_by_penalty():
    tracker = TrustTracker(initial_trust=1.0, flag_penalty=0.3)
    new_trust = tracker.update("agent_1", flagged=True)
    assert new_trust == 0.3


def test_clean_round_increases_trust_by_reward():
    tracker = TrustTracker(initial_trust=0.5, consistency_reward=0.05)
    tracker._trust["agent_1"] = 0.5
    new_trust = tracker.update("agent_1", flagged=False)
    assert new_trust == 0.55


def test_trust_never_exceeds_one():
    tracker = TrustTracker(consistency_reward=0.5)
    tracker._trust["agent_1"] = 0.9
    new_trust = tracker.update("agent_1", flagged=False)
    assert new_trust == 1.0


def test_trust_never_drops_below_min_trust():
    tracker = TrustTracker(flag_penalty=0.3, min_trust=0.1)
    tracker._trust["agent_1"] = 0.2
    new_trust = tracker.update("agent_1", flagged=True)
    assert new_trust == max(0.1, 0.2 * 0.3)  # 0.1 floor wins here
    assert new_trust >= 0.1


def test_trust_persists_across_multiple_updates():
    tracker = TrustTracker(initial_trust=1.0, flag_penalty=0.3, consistency_reward=0.05)
    tracker.update("agent_1", flagged=True)   # 1.0 -> 0.3
    tracker.update("agent_1", flagged=False)  # 0.3 -> 0.35
    assert tracker.get("agent_1") == 0.35


def test_multiple_agents_tracked_independently():
    tracker = TrustTracker()
    tracker.update("agent_1", flagged=True)
    tracker.update("agent_2", flagged=False)
    assert tracker.get("agent_1") < tracker.get("agent_2")


def test_all_scores_returns_everything_tracked():
    tracker = TrustTracker()
    tracker.update("agent_1", flagged=True)
    tracker.update("agent_2", flagged=False)
    scores = tracker.all_scores()
    assert set(scores.keys()) == {"agent_1", "agent_2"}


def test_untouched_agent_not_in_all_scores_but_reads_default():
    tracker = TrustTracker(initial_trust=1.0)
    assert tracker.get("never_touched") == 1.0
    assert "never_touched" not in tracker.all_scores()


# ---------------------------------------------------------------------------
# cross_round_inconsistent
# ---------------------------------------------------------------------------

def test_cross_round_same_answer_not_inconsistent():
    current = "Reasoning... Final answer: Yale"
    previous = "Different reasoning. Final answer: Yale"
    assert not cross_round_inconsistent(current, previous, gold_answer="Yale")


def test_cross_round_different_answer_is_inconsistent():
    current = "Final answer: Duke"
    previous = "Final answer: Yale"
    assert cross_round_inconsistent(current, previous, gold_answer="Yale")


def test_cross_round_paraphrase_of_same_answer_not_inconsistent():
    current = "It was Yale University that won."
    previous = "Final answer: Yale"
    assert not cross_round_inconsistent(current, previous, gold_answer="Yale")


# ---------------------------------------------------------------------------
# compute_flag
# ---------------------------------------------------------------------------

def test_compute_flag_true_if_either_trigger_fires():
    assert compute_flag(rag_flagged=True, inconsistent=False)
    assert compute_flag(rag_flagged=False, inconsistent=True)
    assert compute_flag(rag_flagged=True, inconsistent=True)


def test_compute_flag_false_if_neither_trigger_fires():
    assert not compute_flag(rag_flagged=False, inconsistent=False)
