"""
RAG evidence and contradiction detection.

Two explicit, documented simplifications (see README/PROJECT_BRIEF.md for
the full rationale — repeated briefly here since it's load-bearing for
what this module does and doesn't claim):

  1. Evidence source: FARM's own dataset-provided `source` passage is used
     directly, rather than a live retrieval pipeline. This is an
     oracle-quality upper bound, not a claim that retrieval was built.
  2. Contradiction detection: a lightweight token-overlap heuristic (reusing
     src/metrics/grading.py's normalization), not a dedicated NLI model —
     chosen specifically to avoid a third model's memory footprint
     alongside the two debate LLMs already sharing limited GPU VRAM.
"""

from __future__ import annotations

from src.metrics.grading import extract_final_answer, normalize_answer


def get_evidence(source_passage: str) -> str:
    """
    Oracle evidence lookup. A thin, explicitly-named passthrough rather
    than being silently inlined elsewhere — so every call site is
    self-documenting about using FARM's provided source, and so this is
    the one place a future live-retrieval swap would happen.
    """
    return source_passage


def is_supported_by_evidence(
    response_text: str,
    evidence: str,
    overlap_threshold: float = 0.3,
) -> bool:
    """
    True if the response's claimed answer has meaningful token overlap
    with the evidence passage — a lightweight proxy for "this claim is
    grounded in the evidence," not a real entailment/contradiction check.

    `overlap_threshold` is deliberately lower than grading.py's
    correctness-matching threshold (0.6): evidence passages are much
    longer than a gold answer, so raw overlap fraction is naturally lower
    even for a genuinely supported claim — this is measuring "grounded in
    the evidence" not "is the exact gold answer," a looser bar by design.
    """
    claim = extract_final_answer(response_text)
    norm_claim = normalize_answer(claim)
    norm_evidence = normalize_answer(evidence)

    claim_tokens = set(norm_claim.split())
    evidence_tokens = set(norm_evidence.split())

    if not claim_tokens:
        return False  # an empty/unparseable claim is never "supported"

    overlap = len(claim_tokens & evidence_tokens) / len(claim_tokens)
    return overlap >= overlap_threshold


def check_contradiction(response_text: str, evidence: str, overlap_threshold: float = 0.3) -> bool:
    """
    Returns True if the response should be FLAGGED (i.e. is NOT supported
    by the evidence) — inverse of is_supported_by_evidence, named for
    clarity at call sites that are specifically about flagging.
    """
    return not is_supported_by_evidence(response_text, evidence, overlap_threshold)


def adjust_entropy_for_evidence(
    entropy: float,
    flagged: bool,
    inflation_factor: float = 2.0,
) -> float:
    """
    Evidence-Adjusted entropy: H'(R_j,t) = H(R_j,t) x inflation_factor if
    flagged, else unchanged. Substituted into IGR's denominator in place
    of raw entropy — makes a flagged agent look like a worse communication
    partner even if its raw token-level entropy was low (i.e. even if it
    "sounded confident").
    """
    return entropy * inflation_factor if flagged else entropy
