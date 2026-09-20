import json

import pytest

from src.llm.fake_client import FakeLLMClient
from src.rag.retriever import RetrievedDocument
from src.rag.verification import (
    build_verification_prompt,
    compute_evidence_support,
    parse_verification_output,
)


def _score_json(score, reasoning="ok"):
    return json.dumps({"evidence_support_score": score, "reasoning": reasoning})


# ---------------------------------------------------------------------------
# build_verification_prompt
# ---------------------------------------------------------------------------

def test_build_verification_prompt_includes_claim_and_all_documents():
    docs = [
        RetrievedDocument("d1", "Jupiter", "Jupiter is the largest planet.", 1, 1.0),
        RetrievedDocument("d2", "Saturn", "Saturn has rings.", 2, 0.5),
    ]
    prompt = build_verification_prompt("Saturn is the largest planet", docs)
    assert "Saturn is the largest planet" in prompt
    assert "Jupiter is the largest planet." in prompt
    assert "Saturn has rings." in prompt


# ---------------------------------------------------------------------------
# parse_verification_output
# ---------------------------------------------------------------------------

def test_parse_valid_score():
    score, status = parse_verification_output(_score_json(0.86))
    assert status == "ok"
    assert score == pytest.approx(0.86)


def test_parse_tolerates_surrounding_text():
    raw = "Here is my answer:\n" + _score_json(0.4) + "\nLet me know if you need more."
    score, status = parse_verification_output(raw)
    assert status == "ok"
    assert score == pytest.approx(0.4)


def test_parse_clips_out_of_range_score():
    score, status = parse_verification_output(_score_json(1.4))
    assert status == "ok"
    assert score == 1.0


def test_parse_returns_none_on_garbage():
    score, status = parse_verification_output("not json")
    assert score is None
    assert status == "failed_fallback"


def test_parse_never_returns_a_fabricated_neutral_score_on_failure():
    score, _ = parse_verification_output("garbage")
    assert score != 0.5
    assert score is None


# ---------------------------------------------------------------------------
# compute_evidence_support
# ---------------------------------------------------------------------------

def test_compute_evidence_support_high_confidence_case():
    llm = FakeLLMClient(scripted_responses=[_score_json(0.86, "directly supported")])
    docs = [RetrievedDocument("d1", "Jupiter", "Jupiter is the largest planet.", 1, 1.0)]
    result = compute_evidence_support(llm, "Jupiter is the largest planet", docs)
    assert result["score"] == pytest.approx(0.86)
    assert result["status"] == "ok"
    assert "Jupiter is the largest planet." in result["prompt"]


def test_compute_evidence_support_low_confidence_case():
    llm = FakeLLMClient(scripted_responses=[_score_json(0.12, "contradicted")])
    docs = [RetrievedDocument("d1", "Jupiter", "Jupiter is the largest planet.", 1, 1.0)]
    result = compute_evidence_support(llm, "Saturn is the largest planet", docs)
    assert result["score"] == pytest.approx(0.12)


def test_compute_evidence_support_rejects_empty_documents():
    llm = FakeLLMClient(scripted_responses=[])
    with pytest.raises(ValueError, match="zero documents"):
        compute_evidence_support(llm, "some claim", documents=[])
    assert len(llm.calls) == 0  # must not call the model at all in this case


def test_compute_evidence_support_calls_model_exactly_once():
    llm = FakeLLMClient(scripted_responses=[_score_json(0.5)])
    docs = [RetrievedDocument("d1", "T", "text", 1, 1.0)]
    compute_evidence_support(llm, "claim", docs)
    assert len(llm.calls) == 1
    assert llm.calls[0]["n"] == 1


def test_compute_evidence_support_propagates_failed_parse_as_none_not_fabricated():
    llm = FakeLLMClient(scripted_responses=["I cannot determine this."])
    docs = [RetrievedDocument("d1", "T", "text", 1, 1.0)]
    result = compute_evidence_support(llm, "claim", docs)
    assert result["score"] is None
    assert result["status"] == "failed_fallback"
