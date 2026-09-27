import pandas as pd

from src.digra.orchestration import run_digra_experiments_for_dataset_model
from src.llm.fake_client import FakeLLMClient
from src.utils.checkpoint import RunRegistry
from src.utils.io import read_jsonl

CORRECT_POOL = ["correct A", "correct B", "correct C"]
INCORRECT_POOL = ["wrong A", "wrong B"]


def _df(n=2):
    rows = [
        {
            "question_id": f"nq_{i:04d}", "question": f"Question {i}?",
            "gold_answer": "Yale", "gold_answer_alternatives": [],
            "source": "Yale University won the championship.",
        }
        for i in range(n)
    ]
    return pd.DataFrame(rows)


def _pools(n=2):
    return {
        f"nq_{i:04d}": {"correct_texts": CORRECT_POOL, "incorrect_texts": INCORRECT_POOL}
        for i in range(n)
    }


def test_runs_all_combinations(tmp_path):
    llm = FakeLLMClient(response_fn=lambda p: "Final answer: Yale")
    registry = RunRegistry(tmp_path / "registry.json")
    out_path = tmp_path / "digra.jsonl"

    result = run_digra_experiments_for_dataset_model(
        llm=llm, dataset_key="nq", model_name="llama", variant="digra",
        df=_df(2), pools=_pools(2), setups=["standard"], seeds=[0],
        n_agents=2, n_rounds=1, registry=registry, out_path=out_path,
    )
    assert result["built"] == 2
    assert result["errors"] == 0
    assert len(list(read_jsonl(out_path))) == 2


def test_resume_skips_completed(tmp_path):
    llm1 = FakeLLMClient(response_fn=lambda p: "Final answer: Yale")
    registry_path = tmp_path / "registry.json"
    out_path = tmp_path / "digra.jsonl"
    run_digra_experiments_for_dataset_model(
        llm=llm1, dataset_key="nq", model_name="llama", variant="digra",
        df=_df(2), pools=_pools(2), setups=["standard"], seeds=[0],
        n_agents=2, n_rounds=1, registry=RunRegistry(registry_path), out_path=out_path,
    )

    llm2 = FakeLLMClient(response_fn=lambda p: "Final answer: Yale")
    result = run_digra_experiments_for_dataset_model(
        llm=llm2, dataset_key="nq", model_name="llama", variant="digra",
        df=_df(2), pools=_pools(2), setups=["standard"], seeds=[0],
        n_agents=2, n_rounds=1, registry=RunRegistry(registry_path), out_path=out_path,
    )
    assert result["built"] == 0
    assert result["skipped"] == 2
    assert len(list(read_jsonl(out_path))) == 2  # not duplicated


def test_setup_agent_mismatch_skipped(tmp_path):
    llm = FakeLLMClient(scripted_responses=[])
    result = run_digra_experiments_for_dataset_model(
        llm=llm, dataset_key="nq", model_name="llama", variant="digra",
        df=_df(1), pools=_pools(1), setups=[(2, 1)], seeds=[0],
        n_agents=5, n_rounds=1, registry=RunRegistry(tmp_path / "r.json"), out_path=tmp_path / "d.jsonl",
    )
    assert result["built"] == 0


def test_missing_pool_logged_and_skipped(tmp_path):
    llm = FakeLLMClient(response_fn=lambda p: "Final answer: Yale")
    df = _df(2)
    pools = {"nq_0000": {"correct_texts": CORRECT_POOL, "incorrect_texts": INCORRECT_POOL}}
    result = run_digra_experiments_for_dataset_model(
        llm=llm, dataset_key="nq", model_name="llama", variant="digra",
        df=df, pools=pools, setups=[(1, 1)], seeds=[0],
        n_agents=2, n_rounds=1, registry=RunRegistry(tmp_path / "r.json"), out_path=tmp_path / "d.jsonl",
    )
    assert result["built"] == 1
    assert result["errors"] == 1


def test_rag_variant_requires_source_uses_df_column(tmp_path):
    llm = FakeLLMClient(response_fn=lambda p: "Final answer: Yale")
    result = run_digra_experiments_for_dataset_model(
        llm=llm, dataset_key="nq", model_name="llama", variant="digra_rag",
        df=_df(1), pools=_pools(1), setups=["standard"], seeds=[0],
        n_agents=2, n_rounds=1, registry=RunRegistry(tmp_path / "r.json"), out_path=tmp_path / "d.jsonl",
    )
    assert result["built"] == 1
    assert result["errors"] == 0


def test_calibrate_n_stops_early_and_returns_timing(tmp_path):
    llm = FakeLLMClient(response_fn=lambda p: "Final answer: Yale")
    result = run_digra_experiments_for_dataset_model(
        llm=llm, dataset_key="nq", model_name="llama", variant="digra",
        df=_df(5), pools=_pools(5), setups=["standard"], seeds=[0],
        n_agents=2, n_rounds=1, registry=RunRegistry(tmp_path / "r.json"),
        out_path=tmp_path / "d.jsonl", calibrate_n=2,
    )
    assert result["built"] == 2  # stopped after 2, not all 5
    assert result["calibrated"] is True
    assert "mean_duration_seconds" in result


def test_calibrated_debates_are_marked_done_not_wasted(tmp_path):
    # a calibration run's debates should count for real — resuming after
    # calibration should skip them, not redo them
    registry_path = tmp_path / "r.json"
    out_path = tmp_path / "d.jsonl"

    llm1 = FakeLLMClient(response_fn=lambda p: "Final answer: Yale")
    run_digra_experiments_for_dataset_model(
        llm=llm1, dataset_key="nq", model_name="llama", variant="digra",
        df=_df(5), pools=_pools(5), setups=["standard"], seeds=[0],
        n_agents=2, n_rounds=1, registry=RunRegistry(registry_path),
        out_path=out_path, calibrate_n=2,
    )

    llm2 = FakeLLMClient(response_fn=lambda p: "Final answer: Yale")
    result = run_digra_experiments_for_dataset_model(
        llm=llm2, dataset_key="nq", model_name="llama", variant="digra",
        df=_df(5), pools=_pools(5), setups=["standard"], seeds=[0],
        n_agents=2, n_rounds=1, registry=RunRegistry(registry_path), out_path=out_path,
    )
    assert result["skipped"] == 2
    assert result["built"] == 3  # the remaining 3 of 5


def test_periodic_backup_fires_during_digra_orchestration(tmp_path):
    from src.utils.backup import PeriodicBackup

    pb = PeriodicBackup(every_n=1, local_dest=None, source_dirs=[])
    calls = {"n": 0}
    original = pb.maybe_backup
    def counting():
        calls["n"] += 1
        return original()
    pb.maybe_backup = counting

    llm = FakeLLMClient(response_fn=lambda p: "Final answer: Yale")
    run_digra_experiments_for_dataset_model(
        llm=llm, dataset_key="nq", model_name="llama", variant="digra",
        df=_df(2), pools=_pools(2), setups=["standard"], seeds=[0],
        n_agents=2, n_rounds=1, registry=RunRegistry(tmp_path / "r.json"),
        out_path=tmp_path / "d.jsonl", periodic_backup=pb,
    )
    assert calls["n"] == 2
