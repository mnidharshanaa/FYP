from src.digra.digra_debate import DigraDebateResult, run_digra_debate
from src.llm.fake_client import FakeLLMClient

CORRECT_POOL = ["correct A", "correct B", "correct C"]
INCORRECT_POOL = ["wrong A", "wrong B"]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_invalid_variant_raises_before_any_llm_call():
    llm = FakeLLMClient(scripted_responses=[])
    try:
        run_digra_debate(
            llm=llm, question_id="q1", question="Q", gold_answer="Yale",
            n_agents=2, n_rounds=2, setup="standard", variant="not_a_real_variant",
        )
        assert False, "expected ValueError"
    except ValueError as e:
        assert "variant" in str(e)
    assert len(llm.calls) == 0


def test_rag_variant_requires_source_passage():
    llm = FakeLLMClient(scripted_responses=[])
    try:
        run_digra_debate(
            llm=llm, question_id="q1", question="Q", gold_answer="Yale",
            n_agents=2, n_rounds=2, setup="standard", variant="digra_rag",
            source_passage=None,
        )
        assert False, "expected ValueError"
    except ValueError as e:
        assert "source_passage" in str(e)
    assert len(llm.calls) == 0


def test_memory_variant_also_requires_source_passage():
    # digra_rag_memory still uses the RAG flag as one of its two triggers
    llm = FakeLLMClient(scripted_responses=[])
    try:
        run_digra_debate(
            llm=llm, question_id="q1", question="Q", gold_answer="Yale",
            n_agents=2, n_rounds=2, setup="standard", variant="digra_rag_memory",
            source_passage=None,
        )
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_plain_digra_variant_does_not_require_source_passage():
    llm = FakeLLMClient(response_fn=lambda p: "Final answer: Yale")
    result = run_digra_debate(
        llm=llm, question_id="q1", question="Q", gold_answer="Yale",
        n_agents=2, n_rounds=2, setup="standard", variant="digra",
        source_passage=None,
    )
    assert isinstance(result, DigraDebateResult)


# ---------------------------------------------------------------------------
# DIG (IG-only ablation)
# ---------------------------------------------------------------------------

def test_dig_variant_accepted_and_does_not_require_source_passage():
    llm = FakeLLMClient(response_fn=lambda p: "Final answer: Yale")
    result = run_digra_debate(
        llm=llm, question_id="q1", question="Q", gold_answer="Yale",
        n_agents=2, n_rounds=2, setup="standard", variant="dig",
        source_passage=None,
    )
    assert isinstance(result, DigraDebateResult)
    assert result.variant == "dig"


def test_dig_variant_runs_no_rag_no_memory_fields():
    # dig should behave like plain digra minus the ratio: no rag_flagged,
    # no trust_after_round, on every record (same as plain "digra").
    llm = FakeLLMClient(response_fn=lambda p: "Final answer: Yale")
    result = run_digra_debate(
        llm=llm, question_id="q1", question="Q", gold_answer="Yale",
        n_agents=3, n_rounds=2, setup="standard", variant="dig",
    )
    for agent_hist in result.agent_records:
        for rec in agent_hist:
            assert rec.rag_flagged is None
            assert rec.trust_after_round is None


def test_dig_variant_performs_partner_selection_in_round_2():
    # Round 2 records should have partners_selected populated (selection
    # happened) exactly like digra — the only difference is HOW the
    # winning subset was scored, not whether selection happens at all.
    llm = FakeLLMClient(response_fn=lambda p: "Final answer: Yale")
    result = run_digra_debate(
        llm=llm, question_id="q1", question="Q", gold_answer="Yale",
        n_agents=3, n_rounds=2, setup="standard", variant="dig",
    )
    for agent_hist in result.agent_records:
        round2 = next(r for r in agent_hist if r.round_idx == 2)
        assert round2.partners_selected is not None
        assert round2.igr_score is not None  # holds raw IG for variant="dig"


def test_dig_and_digra_are_the_same_code_path_minus_the_ratio():
    # Both variants must issue the same NUMBER of generate/forced_decode
    # calls for an identical scenario — they differ only in how the
    # already-computed entropies are turned into a selection score, not
    # in how many LLM calls the loop makes. A call-count mismatch here
    # would mean dig accidentally diverged from being "the same loop".
    def make_llm():
        return FakeLLMClient(response_fn=lambda p: "Final answer: Yale")

    llm_dig = make_llm()
    result_dig = run_digra_debate(
        llm=llm_dig, question_id="q1", question="Q", gold_answer="Yale",
        n_agents=3, n_rounds=3, setup="standard", variant="dig", seed=0,
    )
    llm_digra = make_llm()
    result_digra = run_digra_debate(
        llm=llm_digra, question_id="q1", question="Q", gold_answer="Yale",
        n_agents=3, n_rounds=3, setup="standard", variant="digra", seed=0,
    )
    assert result_dig.n_generate_calls == result_digra.n_generate_calls
    assert result_dig.n_forced_decode_calls == result_digra.n_forced_decode_calls


# ---------------------------------------------------------------------------
# Round-1 entropy handling
# ---------------------------------------------------------------------------

def test_standard_setup_round1_uses_fresh_logprobs_no_forced_decode():
    logprobs = [{"yale": -0.1}]
    llm = FakeLLMClient(scripted_responses=[
        ("Final answer: Yale", logprobs), ("Final answer: Yale", logprobs),
    ])
    result = run_digra_debate(
        llm=llm, question_id="q1", question="Q", gold_answer="Yale",
        n_agents=2, n_rounds=1, setup="standard", variant="digra",
    )
    assert result.n_forced_decode_calls == 0
    assert result.agent_records[0][0].entropy > 0  # computed from real logprobs, not 0.0 fallback


def test_seeded_setup_round1_requires_forced_decode_per_agent():
    llm = FakeLLMClient(scripted_responses=[
        ("ignored", [{"a": 0.0}]), ("ignored", [{"a": 0.0}]),
    ])
    result = run_digra_debate(
        llm=llm, question_id="q1", question="Q", gold_answer="Yale",
        n_agents=2, n_rounds=1, setup=(1, 1), variant="digra",
        correct_pool=CORRECT_POOL, incorrect_pool=INCORRECT_POOL,
    )
    # one forced_decode call per agent, purely to establish round-1 entropy
    assert result.n_forced_decode_calls == 2
    assert result.n_generate_calls == 0  # no generation at all for seeded round 1


# ---------------------------------------------------------------------------
# Batching (the actual performance-critical property)
# ---------------------------------------------------------------------------

def test_round2_generation_is_one_batched_call_not_one_per_agent():
    call_log = []

    def resp_fn(p):
        call_log.append("call")
        return "Final answer: A" if len(call_log) % 2 else "Final answer: B"

    llm = FakeLLMClient(response_fn=resp_fn)
    result = run_digra_debate(
        llm=llm, question_id="q1", question="Q", gold_answer="Yale",
        n_agents=3, n_rounds=2, setup="standard", variant="digra",
    )
    generate_batch_calls = [c for c in llm.calls if c["method"] == "generate_batch"]
    assert len(generate_batch_calls) == 1
    assert generate_batch_calls[0]["n"] == 3  # all 3 agents in one call


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

def test_consensus_triggers_early_stop():
    llm = FakeLLMClient(response_fn=lambda p: "Final answer: Yale")
    result = run_digra_debate(
        llm=llm, question_id="q1", question="Q", gold_answer="Yale",
        n_agents=2, n_rounds=5, setup="standard", variant="digra",
    )
    assert result.early_stopped
    assert result.n_rounds_run < 5


def test_no_consensus_runs_all_requested_rounds():
    counter = {"n": 0}

    def resp_fn(p):
        counter["n"] += 1
        return f"Final answer: option{counter['n']}"

    llm = FakeLLMClient(response_fn=resp_fn)
    result = run_digra_debate(
        llm=llm, question_id="q1", question="Q", gold_answer="Yale",
        n_agents=2, n_rounds=3, setup="standard", variant="digra",
    )
    assert result.n_rounds_run == 3


def test_frozen_agent_carried_forward_gets_correct_round_idx():
    # regression test for a real bug caught during manual testing: a frozen
    # agent's carried-forward record was reusing the PREVIOUS round's
    # record object unchanged, producing a duplicate entry still labeled
    # with the old round_idx instead of a fresh one for the current round.
    calls = {"n": 0}

    def resp_fn(p):
        if "Your previous response was" not in p:
            calls["n"] += 1
            return "Final answer: Yale" if calls["n"] == 1 else "Final answer: Duke"
        if "Your previous response was:\nFinal answer: Yale" in p:
            return "Final answer: Yale"  # agent A always repeats -> freezes after round 2
        calls["n"] += 1
        return f"Final answer: Duke{calls['n']}"  # agent B always changes -> never freezes

    llm = FakeLLMClient(response_fn=resp_fn)
    result = run_digra_debate(
        llm=llm, question_id="q1", question="Q", gold_answer="Yale",
        n_agents=2, n_rounds=3, setup="standard", variant="digra",
    )
    agent_a_records = result.agent_records[0]
    round_indices = [r.round_idx for r in agent_a_records]
    assert round_indices == [1, 2, 3]  # no duplicates, no gaps
    assert len(agent_a_records) == 3
    # the carried-forward round-3 record should have no selection info
    assert agent_a_records[2].partners_selected is None


# ---------------------------------------------------------------------------
# Result shape / setup labeling
# ---------------------------------------------------------------------------

def test_setup_label_for_seeded_setup():
    llm = FakeLLMClient(scripted_responses=[("x", [{"a": 0.0}])] * 2)
    result = run_digra_debate(
        llm=llm, question_id="q1", question="Q", gold_answer="Yale",
        n_agents=2, n_rounds=1, setup=(1, 1), variant="digra",
        correct_pool=CORRECT_POOL, incorrect_pool=INCORRECT_POOL,
    )
    assert result.setup == "1,1"


def test_setup_label_for_standard_setup():
    llm = FakeLLMClient(response_fn=lambda p: "Final answer: Yale")
    result = run_digra_debate(
        llm=llm, question_id="q1", question="Q", gold_answer="Yale",
        n_agents=2, n_rounds=1, setup="standard", variant="digra",
    )
    assert result.setup == "standard"


def test_agent_records_shape_matches_n_agents():
    llm = FakeLLMClient(response_fn=lambda p: "Final answer: Yale")
    result = run_digra_debate(
        llm=llm, question_id="q1", question="Q", gold_answer="Yale",
        n_agents=4, n_rounds=1, setup="standard", variant="digra",
    )
    assert len(result.agent_records) == 4


def test_rag_variant_populates_rag_flagged_field():
    llm = FakeLLMClient(response_fn=lambda p: "Final answer: Duke")
    result = run_digra_debate(
        llm=llm, question_id="q1", question="Q", gold_answer="Yale",
        n_agents=2, n_rounds=1, setup="standard", variant="digra_rag",
        source_passage="Yale University won the championship.",
    )
    assert result.agent_records[0][0].rag_flagged is True  # "Duke" not supported


def test_plain_digra_variant_leaves_rag_and_trust_fields_none():
    llm = FakeLLMClient(response_fn=lambda p: "Final answer: Yale")
    result = run_digra_debate(
        llm=llm, question_id="q1", question="Q", gold_answer="Yale",
        n_agents=2, n_rounds=1, setup="standard", variant="digra",
    )
    record = result.agent_records[0][0]
    assert record.rag_flagged is None
    assert record.trust_after_round is None


def test_memory_variant_populates_trust_field():
    llm = FakeLLMClient(response_fn=lambda p: "Final answer: Yale")
    result = run_digra_debate(
        llm=llm, question_id="q1", question="Q", gold_answer="Yale",
        n_agents=2, n_rounds=1, setup="standard", variant="digra_rag_memory",
        source_passage="Yale University won.",
    )
    assert result.agent_records[0][0].trust_after_round is not None
