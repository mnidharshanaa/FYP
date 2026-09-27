"""
Evidence-support scoring for the conditional-RAG "VERIFY" stage. Takes a
claim plus ACTUAL RETRIEVED documents (from src.rag.retriever, never the
model's own parametric memory) and asks the same debate LLM to judge
support conditioned on that retrieved text.

This is legitimate specifically BECAUSE real external text is injected
as context — independence comes from the evidence being retrieved, not
from avoiding the LLM entirely. What would NOT be legitimate is skipping
retrieval and asking "is this claim true?" from the model's own memory;
this module structurally cannot do that, since `compute_evidence_support`
refuses to run with zero documents (see below) rather than silently
falling back to an unconditioned judgment.
"""

from __future__ import annotations

import json
from typing import Optional

from src.llm.client import LLMClient

VERIFICATION_PROMPT_TEMPLATE = """Claim: {claim}

Retrieved evidence:
{evidence_block}

Based ONLY on the retrieved evidence above — not your own general knowledge — rate your confidence from 0.0 to 1.0 that the evidence SUPPORTS the claim. 0.0 means the evidence clearly contradicts the claim or is irrelevant to it. 1.0 means the evidence clearly and directly supports the claim.

Respond with ONLY a JSON object of this exact form, with no other text before or after it:
{{"evidence_support_score": <number between 0.0 and 1.0>, "reasoning": "<one short sentence>"}}"""


def build_verification_prompt(claim: str, documents: list) -> str:
    evidence_block = "\n\n".join(
        f"[{i + 1}] {d.title}: {d.text}" for i, d in enumerate(documents)
    )
    return VERIFICATION_PROMPT_TEMPLATE.format(claim=claim, evidence_block=evidence_block)


def parse_verification_output(raw_text: str) -> tuple:
    """Returns (score: float|None, status: 'ok'|'failed_fallback'). A
    parsing failure returns score=None — NEVER a fabricated neutral 0.5 —
    so the caller (src/reliability/propagation_debate.py) must decide
    explicitly how to treat a failed verification (e.g. leave R
    unchanged and flag it) rather than this silently injecting a
    plausible-looking number that was never actually judged."""
    try:
        start = raw_text.index("{")
        end = raw_text.rindex("}") + 1
        parsed = json.loads(raw_text[start:end])
        score = float(parsed["evidence_support_score"])
        return max(0.0, min(1.0, score)), "ok"
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None, "failed_fallback"


def compute_evidence_support(
    llm: LLMClient, claim: str, documents: list, max_tokens: int = 200,
) -> dict:
    """
    Runs ONE LLM call conditioned on `documents`. Returns
    {"score": float|None, "status": str, "raw_llm_output": str,
    "prompt": str}.

    Raises ValueError on an empty `documents` list rather than silently
    scoring it — "no evidence was retrievable for this claim" is a
    distinct, meaningful case (log retriever_name and
    evidence_support_score=None; do not fall back to an unconditioned
    LLM judgment, which would defeat the entire point of this module)
    that the caller must handle explicitly.
    """
    if not documents:
        raise ValueError(
            "compute_evidence_support called with zero documents. This is a "
            "'no evidence available' case the caller must handle explicitly "
            "(log evidence_support_score=None, leave R unchanged) — it must "
            "never silently fall back to an unconditioned LLM judgment."
        )

    prompt = build_verification_prompt(claim, documents)
    result = llm.generate(prompt, n=1, max_tokens=max_tokens, temperature=0.0, top_p=1.0, top_k=1)[0]
    score, status = parse_verification_output(result.text)
    return {"score": score, "status": status, "raw_llm_output": result.text, "prompt": prompt}
