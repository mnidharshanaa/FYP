from src.utils.io import append_jsonl, read_jsonl, read_jsonl_ids


def test_append_and_read_roundtrip(tmp_path):
    path = tmp_path / "pools.jsonl"
    append_jsonl(path, {"question_id": "q1", "correct_texts": ["a", "b"]})
    append_jsonl(path, {"question_id": "q2", "correct_texts": ["c"]})

    records = list(read_jsonl(path))
    assert len(records) == 2
    assert records[0]["question_id"] == "q1"
    assert records[0]["correct_texts"] == ["a", "b"]
    assert records[1]["question_id"] == "q2"


def test_read_jsonl_missing_file_yields_nothing(tmp_path):
    records = list(read_jsonl(tmp_path / "does_not_exist.jsonl"))
    assert records == []


def test_append_jsonl_creates_parent_dirs(tmp_path):
    path = tmp_path / "nested" / "dir" / "pools.jsonl"
    append_jsonl(path, {"a": 1})
    assert path.exists()


def test_read_jsonl_ids(tmp_path):
    path = tmp_path / "pools.jsonl"
    append_jsonl(path, {"question_id": "q1", "x": 1})
    append_jsonl(path, {"question_id": "q2", "x": 2})
    ids = read_jsonl_ids(path, id_field="question_id")
    assert ids == {"q1", "q2"}


def test_read_jsonl_ids_missing_file_returns_empty_set(tmp_path):
    ids = read_jsonl_ids(tmp_path / "nope.jsonl", id_field="question_id")
    assert ids == set()


def test_append_jsonl_survives_resume_style_reopen(tmp_path):
    path = tmp_path / "pools.jsonl"
    append_jsonl(path, {"question_id": "q1"})
    # simulate a fresh process re-appending after a "disconnect"
    append_jsonl(path, {"question_id": "q2"})
    assert read_jsonl_ids(path, "question_id") == {"q1", "q2"}
