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
        lam=0.5, tau_p=0.6, tau_l=0.3, n_resample=3, theta=0.9,
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
        lam=0.5, tau_p=0.6, tau_l=0.3, n_resample=3, theta=0.9,
    )
    assert result.n_rounds_run >= 1
    assert len(result.agent_reliability_records) > 0
    assert len(result.communication_records) > 0

    selected = [rec for rec in result.communication_records if rec.selected_edge]
    not_selected = [rec for rec in result.communication_records if not rec.selected_edge]
    assert len(selected) > 0  # DIGRA must have selected SOMETHING for someone

    for rec in selected:
        # U is now per-RECEIVER, not per-edge (see propagation_debate.py's
        # module docstring for why) — but it's still logged on every
        # selected edge sharing that receiver, and must be in [0,1].
        assert rec.decision in ("propagate", "verify", "suppress")
        assert rec.quadrant in ("propagate", "verify", "suppress", "low_priority")
        assert 0.0 <= rec.u <= 1.0
        assert 0.0 <= rec.r_source <= 1.0

    for rec in not_selected:
        # A candidate DIGRA never selected is logged for auditability but
        # never gated — no U/R/PS/decision at all.
        assert rec.u is None
        assert rec.r_source is None
        assert rec.ps is None
        assert rec.decision is None
        assert rec.result_correct is None  # never backfilled either


# ---------------------------------------------------------------------------
# Suppressed edges must be excluded from the receiving agent's next prompt.
# Monkeypatch score_and_decide (already exhaustively unit-tested in
# isolation) to force a controlled decision per source agent, so this test
# verifies ORCHESTRATION WIRING specifically, independent of whatever real
# entropy/entailment values a FakeLLMClient run would produce.
# ---------------------------------------------------------------------------

def test_suppressed_source_text_excluded_from_receiver_prompt(monkeypatch):
    def fake_score_and_decide(u, r, lam, tau_p, tau_l, **kwargs):
        from src.reliability.propagation_score import PropagationDecisionRecord
        return PropagationDecisionRecord(
            u=u, r=r, lam=lam, ps=0.9, tau_p=tau_p, tau_l=tau_l,
            decision="suppress", quadrant="suppress",
        )

    llm = FakeLLMClient(response_fn=_constant_response_fn("Final answer: Duke"))
    monkeypatch.setattr(pd_module, "score_and_decide", fake_score_and_decide)

    result = run_propagation_debate(
        llm, "q1", "Who won?", gold_answer="Yale",
        n_agents=3, n_rounds=2, setup="standard",
        lam=0.5, tau_p=0.6, tau_l=0.3, n_resample=2, theta=0.9,
    )

    # Whatever DIGRA selected, the gate always says suppress -> every
    # SELECTED edge must show that (non-selected candidates never reach
    # the gate at all, see the smoke test above for that behavior).
    selected = [rec for rec in result.communication_records if rec.selected_edge]
    assert len(selected) > 0
    for rec in selected:
        assert rec.decision == "suppress"

    # Regardless of exactly how many/which candidates DIGRA selected,
    # nothing should ever survive to the next prompt when the gate always
    # suppresses -> every round-2 per-agent prompt shows no context.
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
# round, evidence verification runs exactly once. Topology is forced via
# monkeypatch (rather than relying on DIGRA's real tie-breaking under
# degenerate zero-entropy fake responses) so the expected dedup count is
# exact and doesn't depend on incidental selection-order behavior.
# ---------------------------------------------------------------------------

def test_rag_deduplicated_across_receivers_for_same_source_and_round(monkeypatch):
    class _FullTopologySelection:
        def __init__(self, candidates):
            self.best_subset = frozenset(candidates)
            self.best_igr = 1.0
            self.all_scores = {frozenset(candidates): 1.0}

    def fake_select_best_partners(candidate_ids, entropy_by_agent, ig_fn, alpha, max_subset_size=None):
        return _FullTopologySelection(candidate_ids)  # every receiver selects BOTH other agents

    def fake_score_and_decide(u, r, lam, tau_p, tau_l, **kwargs):
        from src.reliability.propagation_score import PropagationDecisionRecord
        return PropagationDecisionRecord(
            u=u, r=r, lam=lam, ps=0.3, tau_p=tau_p, tau_l=tau_l,
            decision="verify", quadrant="verify",
        )

    call_count = {"n": 0}
    real_compute_evidence_support = pd_module.compute_evidence_support

    def counting_compute_evidence_support(llm, claim, documents, **kwargs):
        call_count["n"] += 1
        return real_compute_evidence_support(llm, claim, documents, **kwargs)

    monkeypatch.setattr(pd_module, "select_best_partners", fake_select_best_partners)
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
        lam=0.5, tau_p=0.9, tau_l=0.1, n_resample=2, theta=0.9,
        retriever=retriever,
    )

    # With full topology forced, every one of the 3 agents is selected as
    # a source by BOTH other receivers -> 6 selected edges total, but only
    # 3 DISTINCT (source, round) pairs -> exactly 3 real RAG calls AND
    # exactly 3 logged records — a cache HIT must reuse the cached result
    # without appending a second, redundant log record.
    assert call_count["n"] == 3
    assert result.n_rag_calls == 3
    assert len(result.rag_verification_records) == 3


# ---------------------------------------------------------------------------
# No evidence available -> conservative fallback, never fabricated support.
# n_agents=2 -> each receiver has exactly ONE candidate, so selection is
# trivial (no tie-breaking ambiguity to control for here).
# ---------------------------------------------------------------------------

def test_verify_with_no_retriever_falls_back_to_suppress_and_logs_status(monkeypatch):
    def fake_score_and_decide(u, r, lam, tau_p, tau_l, **kwargs):
        from src.reliability.propagation_score import PropagationDecisionRecord
        return PropagationDecisionRecord(
            u=u, r=r, lam=lam, ps=0.3, tau_p=tau_p, tau_l=tau_l,
            decision="verify", quadrant="verify",
        )

    monkeypatch.setattr(pd_module, "score_and_decide", fake_score_and_decide)
    llm = FakeLLMClient(response_fn=_constant_response_fn("Final answer: Duke"))

    result = run_propagation_debate(
        llm, "q1", "Who won?", gold_answer="Yale",
        n_agents=2, n_rounds=2, setup="standard",
        lam=0.5, tau_p=0.9, tau_l=0.1, n_resample=2, theta=0.9,
        retriever=None,  # no evidence source at all
    )

    assert len(result.rag_verification_records) > 0
    for rec in result.rag_verification_records:
        assert rec.evidence_status == "no_evidence_available"
        assert rec.evidence_support_score is None
        assert rec.r_after == rec.r_before  # never fabricated a change
        assert rec.decision_after == "suppress"

    selected = [rec for rec in result.communication_records if rec.selected_edge]
    assert len(selected) > 0
    for comm in selected:
        assert comm.decision == "suppress"  # the fallback, not a fabricated propagate
        assert comm.pre_rag_decision == "verify"  # the ORIGINAL decision is preserved separately
