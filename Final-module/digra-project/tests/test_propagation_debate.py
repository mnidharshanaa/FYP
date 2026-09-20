import json

import pytest

from src.llm.fake_client import FakeLLMClient
from src.rag.retriever import RetrievedDocument, SnapshotRetriever, save_snapshot
import src.reliability.propagation_debate as pd_module
from src.reliability.propagation_debate import run_propagation_debate


def _constant_response_fn(text="Final answer: Yale"):
    return lambda prompt: text


# ---------------------------------------------------------------------------
# Smoke test: runs to completion, correct output shape, no crash.
# response_fn mode -> zero entropy everywhere -> degenerate-but-valid U/R,
# specifically chosen here to test STRUCTURE, not decision differentiation
# (that's already covered exhaustively in test_propagation_score.py and
# test_entailment.py against the pure functions in isolation).
# ---------------------------------------------------------------------------

def test_smoke_full_run_completes_and_shapes_are_correct():
    llm = FakeLLMClient(response_fn=_constant_response_fn())
    result = run_propagation_debate(
        llm, "q1", "Who won the war?", gold_answer="Yale",
        n_agents=3, n_rounds=3, setup="standard",
        lam=0.5, tau_p=0.6, tau_u=0.5, n_resample=3, theta=0.9,
    )
    assert result.question_id == "q1"
    assert result.n_agents == 3
    assert len(result.agent_records) == 3
    # Matches src.digra.digra_debate's OWN established early-stop
    # semantics exactly (this module deliberately mirrors it): the
    # unanimity check only runs AFTER a round_idx's texts are computed,
    # so a debate that's already unanimous from round 1 still executes
    # round_idx=2 once (nothing changes) before the check fires and
    # stops further rounds — n_rounds_run is 2, not 1, here.
    assert result.early_stopped is True
    assert result.n_rounds_run == 2
    for records in result.agent_records:
        assert len(records) == 2


def test_smoke_full_run_without_early_stopping_produces_communication_records():
    # Force non-identical round-1 answers so early stopping doesn't
    # trigger after round 1, exercising the actual communication/decision
    # machinery for at least one full round.
    def response_fn(prompt):
        if "Yale" not in prompt and "Duke" not in prompt:
            # Round 1: no prior context in the prompt yet -> vary by content hash
            return "Final answer: Yale" if "war" in prompt else "Final answer: Duke"
        return "Final answer: Yale"

    llm = FakeLLMClient(response_fn=response_fn)
    result = run_propagation_debate(
        llm, "q1", "Who won the war?", gold_answer="Yale",
        n_agents=3, n_rounds=3, setup="standard",
        lam=0.5, tau_p=0.6, tau_u=0.5, n_resample=3, theta=0.9,
    )
    assert result.n_rounds_run >= 1
    assert len(result.agent_reliability_records) > 0
    assert len(result.communication_records) > 0
    for rec in result.communication_records:
        assert rec.decision in ("propagate", "verify", "suppress")
        assert rec.quadrant in ("propagate", "verify", "suppress", "low_priority")
        assert 0.0 <= rec.u <= 1.0
        assert 0.0 <= rec.r_source <= 1.0


# ---------------------------------------------------------------------------
# Suppressed edges must be excluded from the receiving agent's next prompt.
# Monkeypatch score_and_decide (already exhaustively unit-tested in
# isolation) to force a controlled decision per source agent, so this test
# verifies ORCHESTRATION WIRING specifically, independent of whatever real
# entropy/entailment values a FakeLLMClient run would produce.
# ---------------------------------------------------------------------------

def test_suppressed_source_text_excluded_from_receiver_prompt(monkeypatch):
    def fake_score_and_decide(u, r, lam, tau_p, tau_u, **kwargs):
        # Agent 0 is ALWAYS suppressed as a source; everyone else propagates.
        from src.reliability.propagation_score import PropagationDecisionRecord
        return PropagationDecisionRecord(
            u=u, r=r, lam=lam, ps=0.9, tau_p=tau_p, tau_u=tau_u,
            decision="suppress", quadrant="suppress",
        )

    llm = FakeLLMClient(response_fn=_constant_response_fn("Final answer: Duke"))
    monkeypatch.setattr(pd_module, "score_and_decide", fake_score_and_decide)

    result = run_propagation_debate(
        llm, "q1", "Who won?", gold_answer="Yale",
        n_agents=3, n_rounds=2, setup="standard",
        lam=0.5, tau_p=0.6, tau_u=0.5, n_resample=2, theta=0.9,
    )

    # Every candidate edge should be logged as "suppress" and NOT allowed.
    assert len(result.communication_records) > 0
    for rec in result.communication_records:
        assert rec.decision == "suppress"

    # Inspect the actual round-2 generate() prompt for agent 1 (or any
    # receiver) and confirm agent 0's round-1 response text never appears
    # in another agent's reconsideration prompt.
    round1_text = "Final answer: Duke"
    round2_prompts = [
        call["prompt"] for call in llm.calls
        if call["method"] == "generate" and call["n"] == 1 and "reconsider" in call["prompt"]
    ]
    assert round2_prompts, "expected at least one round-2 per-agent generate() call"
    for prompt in round2_prompts:
        assert "no other agents to consider this round" in prompt


# ---------------------------------------------------------------------------
# RAG must be deduplicated per (source_agent, round): if multiple
# receivers independently reach "verify" for the same source in the same
# round, evidence verification runs exactly once.
# ---------------------------------------------------------------------------

def test_rag_deduplicated_across_receivers_for_same_source_and_round(monkeypatch):
    def fake_score_and_decide(u, r, lam, tau_p, tau_u, **kwargs):
        from src.reliability.propagation_score import PropagationDecisionRecord
        return PropagationDecisionRecord(
            u=u, r=r, lam=lam, ps=0.3, tau_p=tau_p, tau_u=tau_u,
            decision="verify", quadrant="verify",
        )

    call_count = {"n": 0}
    real_compute_evidence_support = pd_module.compute_evidence_support

    def counting_compute_evidence_support(llm, claim, documents, **kwargs):
        call_count["n"] += 1
        return real_compute_evidence_support(llm, claim, documents, **kwargs)

    monkeypatch.setattr(pd_module, "score_and_decide", fake_score_and_decide)
    monkeypatch.setattr(pd_module, "compute_evidence_support", counting_compute_evidence_support)

    snapshot_path = "/tmp/test_rag_dedup_snapshot.json"
    save_snapshot(snapshot_path, {
        "q1": [RetrievedDocument("d1", "T", "Yale won the war.", 1, 1.0)],
    })
    retriever = SnapshotRetriever(snapshot_path)

    llm = FakeLLMClient(response_fn=lambda p: (
        json.dumps({"evidence_support_score": 0.9, "reasoning": "supported"})
        if "Retrieved evidence" in p else "Final answer: Duke"
    ))

    result = run_propagation_debate(
        llm, "q1", "Who won?", gold_answer="Yale",
        n_agents=3, n_rounds=2, setup="standard",
        lam=0.5, tau_p=0.9, tau_u=0.1, n_resample=2, theta=0.9,
        retriever=retriever,
    )

    # 3 receiving agents each independently trigger "verify" for EACH of
    # their 2 candidate sources -> 6 edges total, but only 3 DISTINCT
    # (source, round) pairs exist (agents 0,1,2 each act as a source for
    # the other 2 receivers) -> exactly 3 real RAG calls AND exactly 3
    # logged records — a cache HIT must reuse the cached result without
    # appending a second, redundant log record (a real bug caught here:
    # an earlier version deduplicated the expensive call correctly but
    # still logged once per edge, producing 6 near-duplicate records).
    assert call_count["n"] == 3
    assert result.n_rag_calls == 3
    assert len(result.rag_verification_records) == 3


# ---------------------------------------------------------------------------
# No evidence available -> conservative fallback, never fabricated support.
# ---------------------------------------------------------------------------

def test_verify_with_no_retriever_falls_back_to_suppress_and_logs_status(monkeypatch):
    def fake_score_and_decide(u, r, lam, tau_p, tau_u, **kwargs):
        from src.reliability.propagation_score import PropagationDecisionRecord
        return PropagationDecisionRecord(
            u=u, r=r, lam=lam, ps=0.3, tau_p=tau_p, tau_u=tau_u,
            decision="verify", quadrant="verify",
        )

    monkeypatch.setattr(pd_module, "score_and_decide", fake_score_and_decide)
    llm = FakeLLMClient(response_fn=_constant_response_fn("Final answer: Duke"))

    result = run_propagation_debate(
        llm, "q1", "Who won?", gold_answer="Yale",
        n_agents=2, n_rounds=2, setup="standard",
        lam=0.5, tau_p=0.9, tau_u=0.1, n_resample=2, theta=0.9,
        retriever=None,  # no evidence source at all
    )

    assert len(result.rag_verification_records) > 0
    for rec in result.rag_verification_records:
        assert rec.evidence_status == "no_evidence_available"
        assert rec.evidence_support_score is None
        assert rec.r_after == rec.r_before  # never fabricated a change
        assert rec.decision_after == "suppress"

    for comm in result.communication_records:
        assert comm.decision == "suppress"  # the fallback, not a fabricated propagate
        assert comm.pre_rag_decision == "verify"  # the ORIGINAL decision is preserved separately
