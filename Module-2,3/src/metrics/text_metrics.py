"""
Exact Match (EM) and token-F1, SQuAD-style.

Why this is separate from src/metrics/grading.py
--------------------------------------------------
grading.is_correct() is a lenient containment/overlap heuristic tuned for
short-answer QA gold-vs-prediction judging (used for the response pools
and for MA/MR/IMR/CR, where "was this round's answer right" is a binary
signal). EM/F1 are stricter, standard, independently-reported metrics the
project's own checklist calls for on NQ specifically (Natural Questions is
conventionally reported via EM/F1, not raw accuracy). Both reuse
grading.normalize_answer so "how do we normalize text" is never defined
twice and can't silently drift apart between the two metric families.
"""

from __future__ import annotations

from typing import Optional

from src.metrics.grading import normalize_answer


def exact_match(prediction: str, gold_answer: str, alternatives: Optional[list] = None) -> bool:
    """True if the normalized prediction exactly equals the normalized gold
    answer or any normalized alternative. Stricter than grading.is_correct
    (no containment/overlap fallback) — this is the standard EM definition."""
    norm_pred = normalize_answer(prediction)
    candidates = [gold_answer] + list(alternatives or [])
    return any(norm_pred == normalize_answer(c) for c in candidates if normalize_answer(c))


def _f1(prediction_tokens: list, gold_tokens: list) -> float:
    if not prediction_tokens or not gold_tokens:
        # Both must be non-empty to score any overlap; two empty strings
        # are not treated as a match here (mirrors grading.is_correct's
        # explicit empty-prediction guard — an unparseable answer should
        # never be rewarded).
        return 0.0
    common = {}
    for tok in prediction_tokens:
        common[tok] = common.get(tok, 0)
    pred_counts: dict = {}
    for tok in prediction_tokens:
        pred_counts[tok] = pred_counts.get(tok, 0) + 1
    gold_counts: dict = {}
    for tok in gold_tokens:
        gold_counts[tok] = gold_counts.get(tok, 0) + 1

    num_common = sum(min(c, gold_counts.get(tok, 0)) for tok, c in pred_counts.items())
    if num_common == 0:
        return 0.0
    precision = num_common / len(prediction_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def token_f1(prediction: str, gold_answer: str, alternatives: Optional[list] = None) -> float:
    """Max token-overlap F1 (SQuAD-style, bag-of-words) between the
    prediction and every gold candidate (primary + alternatives), reusing
    grading.normalize_answer for tokenization consistency."""
    pred_tokens = normalize_answer(prediction).split()
    candidates = [gold_answer] + list(alternatives or [])
    best = 0.0
    for candidate in candidates:
        gold_tokens = normalize_answer(candidate).split()
        if not gold_tokens:
            continue
        best = max(best, _f1(pred_tokens, gold_tokens))
    return best
