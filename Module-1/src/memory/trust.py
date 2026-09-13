"""
Persistent trust/memory (this project's second novel contribution).

Two independent triggers feed the same trust score, both at zero extra
inference cost:
  1. Evidence contradiction flag (src/rag/evidence.py's check_contradiction),
     computed from data already generated this round.
  2. Cross-round answer inconsistency: did this agent's extracted answer
     change since the previous round? Computed from data the debate loop
     already produces every round (no resampling, no entailment model —
     see PROJECT_BRIEF.md's discussion of why this is "inspired by," not
     an implementation of, Base Paper 2's bidirectional entailment
     clustering, which would require expensive N-way resampling this
     project deliberately avoids).

Update rule:
  flagged this round (either trigger)  -> T *= flag_penalty
  clean round (neither trigger fired)  -> T += consistency_reward, capped at 1.0
  T never drops below min_trust (a once-wrong agent can later be right again)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.metrics.grading import extract_final_answer, is_correct


@dataclass
class TrustTracker:
    """
    One instance per debate (not per question set) — trust is scoped to
    the agents within a single debate, reset fresh for each new debate.
    """
    initial_trust: float = 1.0
    flag_penalty: float = 0.3
    consistency_reward: float = 0.05
    min_trust: float = 0.1
    _trust: dict = field(default_factory=dict)

    def get(self, agent_id) -> float:
        return self._trust.get(agent_id, self.initial_trust)

    def update(self, agent_id, flagged: bool) -> float:
        """Apply one round's update for `agent_id`. Returns the new trust score."""
        current = self.get(agent_id)
        if flagged:
            new_trust = max(self.min_trust, current * self.flag_penalty)
        else:
            new_trust = min(1.0, current + self.consistency_reward)
        self._trust[agent_id] = new_trust
        return new_trust

    def all_scores(self) -> dict:
        return dict(self._trust)


def cross_round_inconsistent(
    current_response: str,
    previous_response: str,
    gold_answer: str,
    alternatives: list = None,
) -> bool:
    """
    True if the agent's extracted answer appears to have changed between
    rounds — graded via the same normalization used for correctness
    grading (src/metrics/grading.py), so "changed" means a materially
    different claimed answer, not just different wording of the same one.

    Compares the two responses' extracted-answer text directly (not
    against gold_answer) — we care whether the agent is being consistent
    with ITSELF round to round, independent of whether either answer is
    actually correct. `gold_answer`/`alternatives` are accepted for
    interface symmetry with other grading call sites but not used in the
    comparison itself; kept as explicit parameters rather than silently
    dropped, in case a future variant wants gold-aware comparison.
    """
    current_claim = extract_final_answer(current_response)
    previous_claim = extract_final_answer(previous_response)
    # Reuse is_correct's normalization+overlap logic symmetrically: treat
    # the previous claim as the "gold" to check the current claim against.
    return not is_correct(current_claim, previous_claim)


def compute_flag(
    rag_flagged: bool,
    inconsistent: bool,
) -> bool:
    """Combine both triggers — flagged if EITHER fires."""
    return rag_flagged or inconsistent
