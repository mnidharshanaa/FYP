"""
Debate-round prompt templates.

Separate from src/data/prompts.py (pool-building prompts have a different
shape — few-shot examples, not other agents' live responses). Shares the
FINAL_ANSWER_INSTRUCTION constant so grading (src/metrics/grading.py's
extract_final_answer) works identically across both.
"""

from __future__ import annotations

from src.data.prompts import FINAL_ANSWER_INSTRUCTION, build_neutral_prompt

# Re-exported so callers only need to import from src.agents.prompts for
# everything debate-related.
__all__ = ["build_neutral_prompt", "build_debate_round_prompt", "FINAL_ANSWER_INSTRUCTION"]


def build_debate_round_prompt(
    question: str,
    own_previous_response: str,
    other_responses: list,
) -> str:
    """
    Fully-connected round >= 2 prompt: the agent sees its own previous
    response plus every other agent's previous-round response, and is
    asked to reconsider. This is Standard MAD's communication topology —
    DIGRA's dynamic (partner-subset) topology (Module 5) reuses this same
    template but with `other_responses` restricted to the selected subset.
    """
    if not other_responses:
        others_block = "(no other agents to consider this round)"
    else:
        others_block = "\n\n".join(
            f"Agent {i + 1}'s response:\n{resp}"
            for i, resp in enumerate(other_responses)
        )

    return (
        f"Question: {question}\n\n"
        f"Your previous response was:\n{own_previous_response}\n\n"
        f"Other agents' responses:\n\n{others_block}\n\n"
        f"Based on this discussion, reconsider and provide your updated "
        f"response. You may keep your original answer if you still believe "
        f"it is correct, or revise it if the other responses are more "
        f"convincing.\n\n"
        f"{FINAL_ANSWER_INSTRUCTION}"
    )
