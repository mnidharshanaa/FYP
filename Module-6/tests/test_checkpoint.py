import csv

from src.utils.checkpoint import ResultsWriter, RunRegistry, make_run_id


def test_make_run_id_is_order_independent():
    id1 = make_run_id(dataset="nq", model="llama", seed=0)
    id2 = make_run_id(seed=0, dataset="nq", model="llama")
    assert id1 == id2


def test_make_run_id_distinguishes_different_values():
    id1 = make_run_id(dataset="nq", seed=0)
    id2 = make_run_id(dataset="nq", seed=1)
    assert id1 != id2


def test_registry_starts_empty(tmp_path):
    reg = RunRegistry(tmp_path / "registry.json")
    assert len(reg) == 0
    assert not reg.is_done("some_run")


def test_registry_mark_done_and_persist(tmp_path):
    path = tmp_path / "registry.json"
    reg = RunRegistry(path)
    reg.mark_done("run_a", metadata={"accuracy": 0.7})
    assert reg.is_done("run_a")
    assert not reg.is_done("run_b")

    # simulate a fresh process (e.g. after a Kaggle disconnect + rerun)
    reg2 = RunRegistry(path)
    assert reg2.is_done("run_a")
    assert len(reg2) == 1


def test_registry_survives_multiple_marks(tmp_path):
    reg = RunRegistry(tmp_path / "registry.json")
    for i in range(10):
        reg.mark_done(f"run_{i}")
    assert len(reg) == 10
    reg2 = RunRegistry(reg.path)
    assert all(reg2.is_done(f"run_{i}") for i in range(10))


def test_results_writer_writes_header_once(tmp_path):
    path = tmp_path / "results.csv"
    fieldnames = ["dataset", "model", "round", "ma"]

    with ResultsWriter(path, fieldnames) as writer:
        writer.write_row({"dataset": "nq", "model": "llama", "round": 1, "ma": 0.5})

    # reopen in append mode (simulating a resumed run) — header must not repeat
    with ResultsWriter(path, fieldnames) as writer:
        writer.write_row({"dataset": "nq", "model": "llama", "round": 2, "ma": 0.6})

    with path.open() as f:
        reader = list(csv.DictReader(f))

    assert len(reader) == 2
    assert reader[0]["round"] == "1"
    assert reader[1]["round"] == "2"


def test_results_writer_rejects_schema_mismatch(tmp_path):
    path = tmp_path / "results.csv"
    writer = ResultsWriter(path, fieldnames=["a", "b"])
    try:
        raised = False
        try:
            writer.write_row({"a": 1, "c": 2})
        except ValueError:
            raised = True
        assert raised
    finally:
        writer.close()
