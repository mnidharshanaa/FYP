"""
BERTScore-based representative-response quality metric.

This is DELIBERATELY separate from entailment.py's clustering mechanism.
Base 2's own Algorithm 1 (bidirectional entailment clustering) takes an
"NLI classifier: M" as its parameter — clustering there is entailment-
based, matching what src/reliability/entailment.py already implements
(via the same debate LLM, per the frozen design decision). BERTScore, in
that paper, is a DIFFERENT thing: their own real-experiment evaluation
metric for how close an agent's actual output is to an ideal/ground-truth
output (Section IV.A, Table I's BertScore/P/R/F1 columns). This module
adds that second, independent metric — computed against gold_answer, not
used anywhere in how clusters are formed or R is derived.

Runs on CPU by default, deliberately: the debate LLM already occupies the
GPU(s) under tensor_parallel_size, and loading a second embedding model
onto the same device risks OOM. Precision/Recall/F1 over a handful of
short strings per question is fast enough on CPU that this isn't a real
performance trade-off in exchange for that safety.

Model choice: defaults to "distilbert-base-uncased", NOT bert-score's own
default (roberta-large) — a deliberate speed/memory trade-off for running
CPU-side alongside a GPU-bound debate. This measurably correlates less
well with human judgment than roberta-large per the original BERTScore
paper's own benchmarks; if you need the more faithful (slower, heavier)
setting, pass model_type="roberta-large" explicitly.
"""

from __future__ import annotations

from typing import Optional

_SCORER_CACHE: dict = {}


def _get_scorer(model_type: str, device: str):
    """Lazily loads and caches one BERTScorer per (model_type, device) —
    loading the underlying model is expensive and must happen once per
    process, not once per question."""
    key = (model_type, device)
    if key not in _SCORER_CACHE:
        from bert_score import BERTScorer  # local import: only needed if this
                                            # metric is actually used, keeps it
                                            # optional for anyone not using it
        _SCORER_CACHE[key] = BERTScorer(
            model_type=model_type, lang="en", device=device, rescale_with_baseline=False,
        )
    return _SCORER_CACHE[key]


def compute_bertscore(
    candidate: str, reference: str, model_type: str = "distilbert-base-uncased", device: str = "cpu",
) -> dict:
    """Returns {"precision": float, "recall": float, "f1": float} for one
    candidate/reference pair."""
    scorer = _get_scorer(model_type, device)
    precision, recall, f1 = scorer.score([candidate], [reference])
    return {"precision": float(precision[0]), "recall": float(recall[0]), "f1": float(f1[0])}


def compute_bertscore_vs_gold(
    candidate: str,
    gold_answer: str,
    gold_answer_alternatives: Optional[list] = None,
    model_type: str = "distilbert-base-uncased",
    device: str = "cpu",
) -> dict:
    """
    Max-F1-over-alternatives, matching src/metrics/text_metrics.py's
    already-established pattern for EM/F1 (a candidate that matches ANY
    acceptable gold phrasing should score against its best match, not be
    penalized for not matching the primary phrasing specifically).
    Returns the full precision/recall/f1 of whichever reference achieved
    the highest F1, not just the winning F1 alone.
    """
    references = [gold_answer] + list(gold_answer_alternatives or [])
    best = None
    for ref in references:
        result = compute_bertscore(candidate, ref, model_type=model_type, device=device)
        if best is None or result["f1"] > best["f1"]:
            best = result
    return best
