"""
Answer grading.

Why this is its own module
---------------------------
Both the response-pool builder (Appendix B-3 — deciding which of 50 sampled
responses are "correct") and the later debate metrics (MA/MR/IMR/CR — Eq.
1-5, which all reduce to per-round correctness) need the exact same
correctness judgment. Splitting it out means both consumers can never
silently disagree about what "correct" means.

Approach: normalize (lowercase, strip punctuation/articles, collapse
whitespace) both the prediction and every accepted gold phrasing, then
check containment of the normalized gold string within the normalized
prediction. Containment (not exact match) is used because model responses
are typically full sentences ("The answer is Yale.") rather than bare
answers, and short free-text QA gold answers are rarely full sentences.

This is a heuristic, not a solved problem — flagged explicitly here rather
than presented as exact. Before trusting it on a real run, sanity-check
`is_correct` by hand against ~20 real model outputs per dataset (see
project README / step "6.7" in the project plan) before relying on it.
"""

from __future__ import annotations

import re
import string
from typing import Optional

_ARTICLES = {"a", "an", "the"}


def normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation, remove English articles, collapse whitespace."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    tokens = [tok for tok in text.split() if tok not in _ARTICLES]
    return " ".join(tokens).strip()


def is_correct(
    prediction: str,
    gold_answer: str,
    alternatives: Optional[list] = None,
    overlap_threshold: float = 0.6,
) -> bool:
    """
    True if either:
      (a) the normalized gold answer (or any alternative) is contained
          within the normalized prediction, or the prediction is contained
          within it — handles short entity-style gold answers ("Yale")
          appearing inside a full-sentence prediction, and vice versa; or
      (b) normalized token overlap between prediction and a gold candidate
          is >= overlap_threshold — handles full-sentence gold answers
          (common in TruthfulQA) that get paraphrased rather than quoted
          verbatim, where strict containment would produce false negatives.

    Empty/whitespace-only gold candidates are skipped entirely (never
    match anything, avoids false positives from empty-string containment).
    """
    norm_pred = normalize_answer(prediction)
    pred_tokens = set(norm_pred.split())
    candidates = [gold_answer] + list(alternatives or [])

    if not norm_pred:
        # An empty/unparseable prediction can never be judged correct.
        # Without this guard, the bidirectional containment check below
        # would treat "" as a substring of every candidate (norm_pred in
        # norm_candidate is trivially True for empty norm_pred), silently
        # marking a blank or malformed response as "correct" against any
        # gold answer — a real bug caught while building the cross-round
        # consistency check (src/memory/trust.py), which reuses this
        # function and would otherwise have corrupted trust/accuracy
        # signals whenever a response failed to parse.
        return False

    for candidate in candidates:
        norm_candidate = normalize_answer(candidate)
        if not norm_candidate:
            continue

        if norm_candidate in norm_pred or norm_pred in norm_candidate:
            return True

        candidate_tokens = set(norm_candidate.split())
        if candidate_tokens:
            overlap = len(candidate_tokens & pred_tokens) / len(candidate_tokens)
            if overlap >= overlap_threshold:
                return True

    return False


def extract_final_answer(response_text: str, marker: str = "final answer:") -> str:
    """
    If the generation prompt asked the model to end with e.g. 'Final answer: X',
    extract just X for grading; otherwise fall back to the full response text
    (so grading still works on responses that didn't follow the format,
    rather than silently failing to extract anything).
    """
    lower = response_text.lower()
    idx = lower.rfind(marker)
    if idx == -1:
        return response_text.strip()
    return response_text[idx + len(marker):].strip().strip(".").strip()


def compute_em(prediction: str, gold_answer: str, alternatives: Optional[list] = None) -> bool:
    """
    Exact Match (checklist's "EM" column, standard SQuAD-style metric):
    True iff the normalized prediction exactly equals the normalized gold
    answer or any normalized alternative — stricter than `is_correct`
    (which also accepts containment/token-overlap), by design: EM is
    meant to be the strict complement to F1 below, not a duplicate of the
    project's own looser correctness heuristic.
    """
    norm_pred = normalize_answer(prediction)
    candidates = [gold_answer] + list(alternatives or [])
    return any(norm_pred == normalize_answer(c) for c in candidates if normalize_answer(c))


def compute_f1(prediction: str, gold_answer: str, alternatives: Optional[list] = None) -> float:
    """
    Token-overlap F1 (checklist's "F1" column, standard SQuAD-style
    metric), best-of across gold_answer and every alternative — the same
    "try every accepted phrasing, take the best" approach `is_correct`
    already uses for its token-overlap branch, but returning the
    continuous F1 score itself rather than a boolean threshold decision.
    """
    norm_pred = normalize_answer(prediction)
    pred_tokens = norm_pred.split()
    if not pred_tokens:
        return 0.0

    best = 0.0
    for candidate in [gold_answer] + list(alternatives or []):
        gold_tokens = normalize_answer(candidate).split()
        if not gold_tokens:
            continue
        overlap = 0
        gold_counts = {}
        for tok in gold_tokens:
            gold_counts[tok] = gold_counts.get(tok, 0) + 1
        pred_counts = {}
        for tok in pred_tokens:
            pred_counts[tok] = pred_counts.get(tok, 0) + 1
        for tok, cnt in pred_counts.items():
            overlap += min(cnt, gold_counts.get(tok, 0))
        if overlap == 0:
            continue
        precision = overlap / len(pred_tokens)
        recall = overlap / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best = max(best, f1)
    return best
