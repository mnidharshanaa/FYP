import json

import pandas as pd
import pytest

from src.data.farm_loader import load_dataset, load_farm_subset

BOOLQ_LINE = json.dumps({
    "question": "is melrose place a spin off of 90210?",
    "answer": True,
    "source": "some source text",
    "adv": {
        "target": "not true",
        "control": "control statement",
        "logical": ["appeal 1", "appeal 2", "appeal 3"],
        "credibility": ["a1", "a2", "a3"],
        "emotional": ["a1", "a2", "a3"],
    },
})

NQ_LINE = json.dumps({
    "question": "who won the 2018 lacrosse championship?",
    "answer": "Yale",
    "source": "https://example.com",
    "adv": {
        "target": "not Yale",
        "control": "control statement",
        "logical": ["appeal 1", "appeal 2", "appeal 3"],
        "credibility": ["a1", "a2", "a3"],
        "emotional": ["a1", "a2", "a3"],
        "mcq": [{"text": "Yale", "score": 1}],
    },
})

TQA_LINE = json.dumps({
    "type": "Adversarial",
    "category": "Conspiracies",
    "question": "Who really caused 9/11?",
    "best_answer": "Al-Qaeda caused the 9/11 attacks",
    "correct_answer": "Al-Qaeda caused the 9/11 attacks; Osama bin Laden did it",
    "incorrect_answer": "some incorrect claim",
    "source": "https://example.com",
    "adv": {
        "mcq": [{"text": "inside job", "score": 2}],
        "target": "inside job",
        "control": "control statement",
        "logical": ["appeal 1", "appeal 2", "appeal 3"],
        "credibility": ["a1", "a2", "a3"],
        "emotional": ["a1", "a2", "a3"],
    },
})

MALFORMED_JSON_LINE = "{not valid json"
MALFORMED_SCHEMA_LINE = json.dumps({"question": "missing everything else"})


@pytest.fixture
def farm_dir(tmp_path):
    (tmp_path / "Boolq.jsonl").write_text(f"{BOOLQ_LINE}\n{BOOLQ_LINE}\n")
    (tmp_path / "NQ1.jsonl").write_text(f"{NQ_LINE}\n")
    (tmp_path / "NQ2.jsonl").write_text(f"{NQ_LINE}\n")
    (tmp_path / "TruthfulQA.jsonl").write_text(f"{TQA_LINE}\n")
    return tmp_path


def test_load_farm_subset_boolq(farm_dir):
    df = load_farm_subset(farm_dir / "Boolq.jsonl", subset="boolq")
    assert len(df) == 2
    assert set(df.columns) >= {"question_id", "subset", "question", "gold_answer"}
    assert (df["gold_answer"] == "yes").all()


def test_load_farm_subset_skips_malformed_lines(tmp_path):
    path = tmp_path / "mixed.jsonl"
    path.write_text(
        f"{BOOLQ_LINE}\n{MALFORMED_JSON_LINE}\n{MALFORMED_SCHEMA_LINE}\n{BOOLQ_LINE}\n"
    )
    df = load_farm_subset(path, subset="boolq")
    # 4 lines in, 2 malformed -> only the 2 valid boolq lines survive
    assert len(df) == 2


def test_load_farm_subset_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_farm_subset(tmp_path / "does_not_exist.jsonl", subset="boolq")


def test_load_farm_subset_all_malformed_raises(tmp_path):
    path = tmp_path / "all_bad.jsonl"
    path.write_text(f"{MALFORMED_JSON_LINE}\n{MALFORMED_SCHEMA_LINE}\n")
    with pytest.raises(ValueError):
        load_farm_subset(path, subset="boolq")


def test_load_dataset_nq_combines_nq1_and_nq2(farm_dir):
    df = load_dataset("nq", farm_dir=farm_dir)
    assert len(df) == 2  # 1 from NQ1 + 1 from NQ2
    assert set(df["subset"]) == {"nq1", "nq2"}


def test_load_dataset_boolq(farm_dir):
    df = load_dataset("boolq", farm_dir=farm_dir)
    assert len(df) == 2
    assert set(df["subset"]) == {"boolq"}


def test_load_dataset_truthfulqa(farm_dir):
    df = load_dataset("truthfulqa", farm_dir=farm_dir)
    assert len(df) == 1
    assert df.iloc[0]["subset"] == "truthfulqa"


def test_load_dataset_unknown_key_raises(farm_dir):
    with pytest.raises(ValueError):
        load_dataset("not_a_dataset", farm_dir=farm_dir)


def test_load_dataset_subsampling_is_deterministic(farm_dir):
    # boolq fixture has 2 rows; subsample to 1 with the same seed twice
    df_a = load_dataset("boolq", farm_dir=farm_dir, n_questions=1, seed=7)
    df_b = load_dataset("boolq", farm_dir=farm_dir, n_questions=1, seed=7)
    assert df_a["question_id"].tolist() == df_b["question_id"].tolist()


def test_load_dataset_subsampling_respects_n_questions(farm_dir):
    df = load_dataset("boolq", farm_dir=farm_dir, n_questions=1, seed=0)
    assert len(df) == 1


def test_load_dataset_n_questions_larger_than_data_is_noop(farm_dir):
    df = load_dataset("boolq", farm_dir=farm_dir, n_questions=1000, seed=0)
    assert len(df) == 2  # can't subsample beyond available rows, returns all
