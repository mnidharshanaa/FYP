import pandas as pd

from src.llm.fake_client import FakeLLMClient
from src.reliability.orchestration import run_propagation_control_for_dataset_model
from src.utils.checkpoint import RunRegistry
from src.utils.io import read_jsonl


def _tiny_df():
    return pd.DataFrame([
        {"question_id": "nq_0000", "question": "Who won?", "gold_answer": "Yale",
         "gold_answer_alternatives": []},
    ])


def _default_kwargs():
    return dict(
        n_agents=3, n_rounds=2, lam=0.5, tau_p=0.6, tau_l=0.3, n_resample=2, theta=0.9,
    )


def test_writes_one_record_per_question(tmp_path):
    llm = FakeLLMClient(response_fn=lambda p: "Final answer: Yale")
    registry = RunRegistry(tmp_path / "registry.json")
    out_path = tmp_path / "nq_llama_propagation_control.jsonl"

    counts = run_propagation_control_for_dataset_model(
        llm=llm, dataset_key="nq", model_name="llama", df=_tiny_df(),
        seeds=[0], registry=registry, out_path=out_path, **_default_kwargs(),
    )
    assert counts == {"built": 1, "skipped": 0, "errors": 0}

    records = list(read_jsonl(out_path))
    assert len(records) == 1
    assert records[0]["question_id"] == "nq_0000"
    assert "agent_reliability_records" in records[0]
    assert "communication_records" in records[0]
    assert "rag_verification_records" in records[0]


def test_skips_already_completed_via_registry(tmp_path):
    llm = FakeLLMClient(response_fn=lambda p: "Final answer: Yale")
    registry = RunRegistry(tmp_path / "registry.json")
    out_path = tmp_path / "out.jsonl"

    run_propagation_control_for_dataset_model(
        llm=llm, dataset_key="nq", model_name="llama", df=_tiny_df(),
        seeds=[0], registry=registry, out_path=out_path, **_default_kwargs(),
    )

    llm2 = FakeLLMClient(scripted_responses=[])
    counts = run_propagation_control_for_dataset_model(
        llm=llm2, dataset_key="nq", model_name="llama", df=_tiny_df(),
        seeds=[0], registry=registry, out_path=out_path, **_default_kwargs(),
    )
    assert counts == {"built": 0, "skipped": 1, "errors": 0}


def test_continues_after_a_question_errors(tmp_path, monkeypatch):
    import src.reliability.orchestration as orch_module

    calls = {"n": 0}
    real_run = orch_module.run_propagation_debate

    def flaky_run(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated failure")
        return real_run(*args, **kwargs)

    monkeypatch.setattr(orch_module, "run_propagation_debate", flaky_run)

    llm = FakeLLMClient(response_fn=lambda p: "Final answer: Yale")
    registry = RunRegistry(tmp_path / "registry.json")
    df = pd.DataFrame([
        {"question_id": "q1", "question": "Q1?", "gold_answer": "Yale", "gold_answer_alternatives": []},
        {"question_id": "q2", "question": "Q2?", "gold_answer": "Yale", "gold_answer_alternatives": []},
    ])
    counts = run_propagation_control_for_dataset_model(
        llm=llm, dataset_key="nq", model_name="llama", df=df,
        seeds=[0], registry=registry, out_path=tmp_path / "out.jsonl", **_default_kwargs(),
    )
    assert counts["errors"] == 1
    assert counts["built"] == 1


def test_different_lambda_values_are_independent_run_ids(tmp_path):
    # Same question/seed/model, different lam -> both should run (lam
    # must be part of the run_id, or a second lambda sweep value would
    # wrongly be skipped as "already done").
    llm = FakeLLMClient(response_fn=lambda p: "Final answer: Yale")
    registry = RunRegistry(tmp_path / "registry.json")

    counts_a = run_propagation_control_for_dataset_model(
        llm=llm, dataset_key="nq", model_name="llama", df=_tiny_df(), seeds=[0],
        n_agents=3, n_rounds=2, lam=0.0, tau_p=0.6, tau_l=0.3, n_resample=2, theta=0.9,
        registry=registry, out_path=tmp_path / "lam0.jsonl",
    )
    counts_b = run_propagation_control_for_dataset_model(
        llm=llm, dataset_key="nq", model_name="llama", df=_tiny_df(), seeds=[0],
        n_agents=3, n_rounds=2, lam=1.0, tau_p=0.6, tau_l=0.3, n_resample=2, theta=0.9,
        registry=registry, out_path=tmp_path / "lam1.jsonl",
    )
    assert counts_a["built"] == 1
    assert counts_b["built"] == 1


# ---------------------------------------------------------------------------
# Setup sweep — "standard" + controlled-correctness conditions (Q1's
# frozen decision: add, don't replace).
# ---------------------------------------------------------------------------

def test_default_setups_is_standard_only(tmp_path):
    llm = FakeLLMClient(response_fn=lambda p: "Final answer: Yale")
    registry = RunRegistry(tmp_path / "registry.json")
    counts = run_propagation_control_for_dataset_model(
        llm=llm, dataset_key="nq", model_name="llama", df=_tiny_df(),
        seeds=[0], registry=registry, out_path=tmp_path / "out.jsonl", **_default_kwargs(),
    )
    assert counts["built"] == 1  # exactly one run: "standard" only, no setups given


def test_controlled_setup_without_pool_is_skipped_as_error(tmp_path):
    # Requesting a controlled-correctness setup with NO pools provided
    # must be logged as an error and skipped, never silently run as if
    # it were "standard" (that would corrupt exactly the controlled
    # initial-correctness condition the setup exists to guarantee).
    llm = FakeLLMClient(scripted_responses=[])
    registry = RunRegistry(tmp_path / "registry.json")
    counts = run_propagation_control_for_dataset_model(
        llm=llm, dataset_key="nq", model_name="llama", df=_tiny_df(), seeds=[0],
        n_agents=3, n_rounds=2, lam=0.5, tau_p=0.6, tau_l=0.3, n_resample=2, theta=0.9,
        registry=registry, out_path=tmp_path / "out.jsonl",
        setups=[(1, 2)],  # controlled, no pools= given
    )
    assert counts == {"built": 0, "skipped": 0, "errors": 1}


def test_controlled_setup_with_pool_runs_successfully(tmp_path):
    llm = FakeLLMClient(response_fn=lambda p: "Final answer: Yale")
    registry = RunRegistry(tmp_path / "registry.json")
    pools = {"nq_0000": {"correct_texts": ["Final answer: Yale"] * 5,
                          "incorrect_texts": ["Final answer: Duke"] * 5}}
    counts = run_propagation_control_for_dataset_model(
        llm=llm, dataset_key="nq", model_name="llama", df=_tiny_df(), seeds=[0],
        n_agents=3, n_rounds=2, lam=0.5, tau_p=0.6, tau_l=0.3, n_resample=2, theta=0.9,
        registry=registry, out_path=tmp_path / "out.jsonl",
        setups=[(1, 2)], pools=pools,
    )
    assert counts == {"built": 1, "skipped": 0, "errors": 0}


def test_setups_sweep_adds_conditions_rather_than_replacing_standard(tmp_path):
    llm = FakeLLMClient(response_fn=lambda p: "Final answer: Yale")
    registry = RunRegistry(tmp_path / "registry.json")
    pools = {"nq_0000": {"correct_texts": ["Final answer: Yale"] * 5,
                          "incorrect_texts": ["Final answer: Duke"] * 5}}
    counts = run_propagation_control_for_dataset_model(
        llm=llm, dataset_key="nq", model_name="llama", df=_tiny_df(), seeds=[0],
        n_agents=3, n_rounds=2, lam=0.5, tau_p=0.6, tau_l=0.3, n_resample=2, theta=0.9,
        registry=registry, out_path=tmp_path / "out.jsonl",
        setups=["standard", (1, 2)], pools=pools,
    )
    assert counts["built"] == 2  # both "standard" AND the controlled condition ran


def test_mismatched_setup_sum_is_skipped_not_run(tmp_path):
    # (2, 2) sums to 4, not n_agents=3 -> must be skipped entirely (same
    # convention as src.digra.orchestration), not run with wrong counts.
    llm = FakeLLMClient(scripted_responses=[])
    registry = RunRegistry(tmp_path / "registry.json")
    counts = run_propagation_control_for_dataset_model(
        llm=llm, dataset_key="nq", model_name="llama", df=_tiny_df(), seeds=[0],
        n_agents=3, n_rounds=2, lam=0.5, tau_p=0.6, tau_l=0.3, n_resample=2, theta=0.9,
        registry=registry, out_path=tmp_path / "out.jsonl",
        setups=[(2, 2)],
    )
    assert counts == {"built": 0, "skipped": 0, "errors": 0}


# ---------------------------------------------------------------------------
# bypass_reliability_gate passthrough (Condition B)
# ---------------------------------------------------------------------------

def test_bypass_reliability_gate_is_a_distinct_run_id(tmp_path):
    llm = FakeLLMClient(response_fn=lambda p: "Final answer: Yale")
    registry = RunRegistry(tmp_path / "registry.json")

    counts_gated = run_propagation_control_for_dataset_model(
        llm=llm, dataset_key="nq", model_name="llama", df=_tiny_df(), seeds=[0],
        n_agents=3, n_rounds=2, lam=0.5, tau_p=0.6, tau_l=0.3, n_resample=2, theta=0.9,
        registry=registry, out_path=tmp_path / "gated.jsonl", bypass_reliability_gate=False,
    )
    counts_bypassed = run_propagation_control_for_dataset_model(
        llm=llm, dataset_key="nq", model_name="llama", df=_tiny_df(), seeds=[0],
        n_agents=3, n_rounds=2, lam=0.5, tau_p=0.6, tau_l=0.3, n_resample=2, theta=0.9,
        registry=registry, out_path=tmp_path / "bypassed.jsonl", bypass_reliability_gate=True,
    )
    assert counts_gated["built"] == 1
    assert counts_bypassed["built"] == 1  # NOT skipped as a duplicate of the gated run
