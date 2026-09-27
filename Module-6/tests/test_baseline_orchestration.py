import json

import pandas as pd

from src.baselines.orchestration import run_baselines_for_dataset_model
from src.llm.fake_client import FakeLLMClient
from src.utils.checkpoint import RunRegistry
from src.utils.io import read_jsonl


def _tiny_df():
    return pd.DataFrame([
        {"question_id": "nq_0000", "question": "Who won?", "gold_answer": "Yale",
         "gold_answer_alternatives": []},
        {"question_id": "nq_0001", "question": "Who lost?", "gold_answer": "Duke",
         "gold_answer_alternatives": []},
    ])


def test_run_baselines_writes_one_record_per_question(tmp_path):
    llm = FakeLLMClient(scripted_responses=["Final answer: Yale"] * 10)
    registry = RunRegistry(tmp_path / "registry.json")
    out_path = tmp_path / "nq_llama_cot_sc.jsonl"

    counts = run_baselines_for_dataset_model(
        llm=llm, dataset_key="nq", model_name="llama", df=_tiny_df(),
        seeds=[0], n_samples_max=5, sc_ks=[1, 3, 5],
        registry=registry, out_path=out_path,
    )

    assert counts == {"built": 2, "skipped": 0, "errors": 0}
    records = list(read_jsonl(out_path))
    assert len(records) == 2
    assert {r["question_id"] for r in records} == {"nq_0000", "nq_0001"}
    for r in records:
        assert set(r["votes"].keys()) == {"1", "3", "5"}
        assert len(r["samples"]) == 5


def test_run_baselines_skips_already_completed_via_registry(tmp_path):
    llm = FakeLLMClient(scripted_responses=["Final answer: Yale"] * 5)
    registry = RunRegistry(tmp_path / "registry.json")
    out_path = tmp_path / "out.jsonl"

    df = _tiny_df().iloc[:1]
    run_baselines_for_dataset_model(
        llm=llm, dataset_key="nq", model_name="llama", df=df,
        seeds=[0], n_samples_max=5, sc_ks=[5],
        registry=registry, out_path=out_path,
    )

    # Second run with a fresh (but exhausted) client — should be skipped
    # entirely, so no further generate() calls happen.
    llm2 = FakeLLMClient(scripted_responses=[])
    counts = run_baselines_for_dataset_model(
        llm=llm2, dataset_key="nq", model_name="llama", df=df,
        seeds=[0], n_samples_max=5, sc_ks=[5],
        registry=registry, out_path=out_path,
    )
    assert counts == {"built": 0, "skipped": 1, "errors": 0}


def test_run_baselines_records_duration_in_registry_metadata(tmp_path):
    llm = FakeLLMClient(scripted_responses=["Final answer: Yale"] * 5)
    registry_path = tmp_path / "registry.json"
    registry = RunRegistry(registry_path)
    out_path = tmp_path / "out.jsonl"

    run_baselines_for_dataset_model(
        llm=llm, dataset_key="nq", model_name="llama", df=_tiny_df().iloc[:1],
        seeds=[0], n_samples_max=5, sc_ks=[5],
        registry=registry, out_path=out_path,
    )

    with registry_path.open() as f:
        data = json.load(f)
    assert len(data) == 1
    (metadata,) = data.values()
    assert "duration_seconds" in metadata
    assert metadata["duration_seconds"] >= 0


def test_run_baselines_continues_after_a_question_errors(tmp_path, monkeypatch):
    import src.baselines.orchestration as orch_module

    calls = {"n": 0}

    def flaky_run_cot_sc(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated backend failure")
        return real_run_cot_sc(*args, **kwargs)

    real_run_cot_sc = orch_module.run_cot_sc
    monkeypatch.setattr(orch_module, "run_cot_sc", flaky_run_cot_sc)

    llm = FakeLLMClient(scripted_responses=["Final answer: Yale"] * 5)
    registry = RunRegistry(tmp_path / "registry.json")
    out_path = tmp_path / "out.jsonl"

    counts = run_baselines_for_dataset_model(
        llm=llm, dataset_key="nq", model_name="llama", df=_tiny_df(),
        seeds=[0], n_samples_max=5, sc_ks=[5],
        registry=registry, out_path=out_path,
    )
    assert counts["errors"] == 1
    assert counts["built"] == 1
