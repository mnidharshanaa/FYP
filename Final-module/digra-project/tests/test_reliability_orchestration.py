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
        n_agents=3, n_rounds=2, lam=0.5, tau_p=0.6, tau_u=0.5, n_resample=2, theta=0.9,
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
        n_agents=3, n_rounds=2, lam=0.0, tau_p=0.6, tau_u=0.5, n_resample=2, theta=0.9,
        registry=registry, out_path=tmp_path / "lam0.jsonl",
    )
    counts_b = run_propagation_control_for_dataset_model(
        llm=llm, dataset_key="nq", model_name="llama", df=_tiny_df(), seeds=[0],
        n_agents=3, n_rounds=2, lam=1.0, tau_p=0.6, tau_u=0.5, n_resample=2, theta=0.9,
        registry=registry, out_path=tmp_path / "lam1.jsonl",
    )
    assert counts_a["built"] == 1
    assert counts_b["built"] == 1
