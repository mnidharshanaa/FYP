import pandas as pd

from src.baselines.mad_variant_orchestration import run_mad_variant_for_dataset_model
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


def test_run_sparse_variant_writes_one_record_per_question(tmp_path):
    llm = FakeLLMClient(scripted_responses=["Final answer: Yale"] * 200)
    registry = RunRegistry(tmp_path / "registry.json")
    out_path = tmp_path / "nq_llama_mad_sparse_half_agents3.jsonl"

    counts = run_mad_variant_for_dataset_model(
        llm=llm, variant="mad_sparse_half", dataset_key="nq", model_name="llama",
        df=_tiny_df(), seeds=[0], n_agents=3, n_rounds=3,
        registry=registry, out_path=out_path, sparse_degree=0.5,
    )

    assert counts == {"built": 2, "skipped": 0, "errors": 0}
    records = list(read_jsonl(out_path))
    assert len(records) == 2
    for r in records:
        assert r["setup"] == "standard"
        assert r["n_agents"] == 3
        assert len(r["agent_responses"]) == 3


def test_run_random_variant_writes_one_record_per_question(tmp_path):
    llm = FakeLLMClient(scripted_responses=["Final answer: Yale"] * 200)
    registry = RunRegistry(tmp_path / "registry.json")
    out_path = tmp_path / "nq_llama_mad_random_agents3.jsonl"

    counts = run_mad_variant_for_dataset_model(
        llm=llm, variant="mad_random", dataset_key="nq", model_name="llama",
        df=_tiny_df(), seeds=[0], n_agents=3, n_rounds=3,
        registry=registry, out_path=out_path,
    )
    assert counts == {"built": 2, "skipped": 0, "errors": 0}


def test_run_mad_variant_skips_already_completed_via_registry(tmp_path):
    llm = FakeLLMClient(scripted_responses=["Final answer: Yale"] * 100)
    registry = RunRegistry(tmp_path / "registry.json")
    out_path = tmp_path / "out.jsonl"
    df = _tiny_df().iloc[:1]

    run_mad_variant_for_dataset_model(
        llm=llm, variant="mad_sparse_half", dataset_key="nq", model_name="llama",
        df=df, seeds=[0], n_agents=3, n_rounds=3,
        registry=registry, out_path=out_path, sparse_degree=0.5,
    )

    llm2 = FakeLLMClient(scripted_responses=[])
    counts = run_mad_variant_for_dataset_model(
        llm=llm2, variant="mad_sparse_half", dataset_key="nq", model_name="llama",
        df=df, seeds=[0], n_agents=3, n_rounds=3,
        registry=registry, out_path=out_path, sparse_degree=0.5,
    )
    assert counts == {"built": 0, "skipped": 1, "errors": 0}


def test_run_mad_variant_different_variants_are_independent_run_ids(tmp_path):
    # Same question/seed/model, different variant -> both should run (the
    # registry key must include variant, or the second would wrongly skip).
    llm = FakeLLMClient(scripted_responses=["Final answer: Yale"] * 200)
    registry = RunRegistry(tmp_path / "registry.json")
    df = _tiny_df().iloc[:1]

    counts_sparse = run_mad_variant_for_dataset_model(
        llm=llm, variant="mad_sparse_half", dataset_key="nq", model_name="llama",
        df=df, seeds=[0], n_agents=3, n_rounds=3,
        registry=registry, out_path=tmp_path / "sparse.jsonl", sparse_degree=0.5,
    )
    counts_random = run_mad_variant_for_dataset_model(
        llm=llm, variant="mad_random", dataset_key="nq", model_name="llama",
        df=df, seeds=[0], n_agents=3, n_rounds=3,
        registry=registry, out_path=tmp_path / "random.jsonl",
    )
    assert counts_sparse["built"] == 1
    assert counts_random["built"] == 1


def test_run_mad_variant_continues_after_a_question_errors(tmp_path, monkeypatch):
    import src.baselines.mad_variant_orchestration as orch_module

    calls = {"n": 0}
    real_run = orch_module.run_mad_variant_debate

    def flaky_run(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated backend failure")
        return real_run(*args, **kwargs)

    monkeypatch.setattr(orch_module, "run_mad_variant_debate", flaky_run)

    llm = FakeLLMClient(scripted_responses=["Final answer: Yale"] * 200)
    registry = RunRegistry(tmp_path / "registry.json")
    out_path = tmp_path / "out.jsonl"

    counts = run_mad_variant_for_dataset_model(
        llm=llm, variant="mad_sparse_half", dataset_key="nq", model_name="llama",
        df=_tiny_df(), seeds=[0], n_agents=3, n_rounds=3,
        registry=registry, out_path=out_path, sparse_degree=0.5,
    )
    assert counts["errors"] == 1
    assert counts["built"] == 1
