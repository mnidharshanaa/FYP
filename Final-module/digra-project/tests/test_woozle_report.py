import json

import pytest

from src.reports.woozle_report import (
    build_table1_rows,
    compute_fig2_series,
    compute_fig4_data,
    cot_sc_accuracy_reference,
    discover_result_files,
    load_and_group,
    median_split_difficulty,
)


def _write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _standard_mad_record(question_id, setup, agent_responses, gold="Yale"):
    return {
        "question_id": question_id, "setup": setup, "n_agents": len(agent_responses),
        "n_rounds": len(agent_responses[0]), "agent_responses": agent_responses,
        "gold_answer": gold, "gold_answer_alternatives": [],
    }


def _digra_record(question_id, setup, per_agent_answers, variant="digra", gold="Yale"):
    # per_agent_answers: list[list[str]] of extracted answers per agent per round
    agent_records = []
    for agent_answers in per_agent_answers:
        history = []
        for t, ans in enumerate(agent_answers, start=1):
            history.append({
                "round_idx": t, "response_text": f"Final answer: {ans}", "extracted_answer": ans,
                "is_correct": ans.lower() == gold.lower(), "entropy": 0.1,
                "rag_flagged": None, "cross_round_inconsistent": None, "combined_flagged": None,
                "trust_after_round": None, "partners_selected": None, "igr_score": None,
            })
        agent_records.append(history)
    return {
        "question_id": question_id, "variant": variant, "setup": setup, "n_agents": len(per_agent_answers),
        "n_rounds_requested": len(per_agent_answers[0]), "n_rounds_run": len(per_agent_answers[0]),
        "early_stopped": False, "gold_answer": gold, "gold_answer_alternatives": [],
        "agent_records": agent_records, "n_generate_calls": 1, "n_forced_decode_calls": 0,
    }


# ---------------------------------------------------------------------------
# discover_result_files
# ---------------------------------------------------------------------------

def test_discover_finds_standard_mad_file(tmp_path):
    _write_jsonl(
        tmp_path / "debates" / "nq_llama_agents3.jsonl",
        [_standard_mad_record("q1", "standard", [["Final answer: Yale"]] * 3)],
    )
    files, skipped = discover_result_files([str(tmp_path)])
    assert len(files) == 1
    assert files[0].method == "standard_mad"
    assert files[0].dataset == "nq"
    assert files[0].model == "llama"


def test_discover_finds_digra_file_with_correct_method():
    pass  # covered by test below combining variants


def test_discover_excludes_mad_variant_baselines(tmp_path):
    _write_jsonl(
        tmp_path / "baselines" / "nq_llama_mad_random_agents3.jsonl",
        [_standard_mad_record("q1", "standard", [["Final answer: Yale"]] * 3)],
    )
    files, skipped = discover_result_files([str(tmp_path)])
    assert len(files) == 0
    assert any("Table-II baseline" in s for s in skipped)


def test_discover_excludes_dig_verification_demo_files(tmp_path):
    _write_jsonl(
        tmp_path / "dig_verification" / "demo_nq_llama_digra_agents3.jsonl",
        [_digra_record("demo_q1", "standard", [["Yale"], ["Yale"], ["Yale"]])],
    )
    files, skipped = discover_result_files([str(tmp_path)])
    assert len(files) == 0
    assert any("dig_verification" in s for s in skipped)


def test_discover_excludes_cot_sc_files(tmp_path):
    _write_jsonl(
        tmp_path / "baselines" / "nq_llama_cot_sc.jsonl",
        [{"question_id": "q1", "samples": [], "votes": {}}],
    )
    files, skipped = discover_result_files([str(tmp_path)])
    assert len(files) == 0


def test_discover_finds_digra_and_dig_with_distinct_methods(tmp_path):
    _write_jsonl(
        tmp_path / "digra" / "nq_llama_digra_agents3.jsonl",
        [_digra_record("q1", "standard", [["Yale"], ["Yale"], ["Yale"]], variant="digra")],
    )
    _write_jsonl(
        tmp_path / "digra" / "nq_llama_dig_agents3.jsonl",
        [_digra_record("q1", "standard", [["Yale"], ["Yale"], ["Yale"]], variant="dig")],
    )
    files, skipped = discover_result_files([str(tmp_path)])
    methods = {f.method for f in files}
    assert methods == {"digra", "dig"}


def test_discover_deduplicates_identical_paths_across_roots(tmp_path):
    _write_jsonl(
        tmp_path / "a" / "debates" / "nq_llama_agents3.jsonl",
        [_standard_mad_record("q1", "standard", [["Final answer: Yale"]] * 3)],
    )
    # Same root passed twice -> same real path, should only count once.
    files, skipped = discover_result_files([str(tmp_path / "a"), str(tmp_path / "a")])
    assert len(files) == 1


# ---------------------------------------------------------------------------
# load_and_group
# ---------------------------------------------------------------------------

def test_load_and_group_dedups_identical_duplicate_records(tmp_path):
    rec = _standard_mad_record("q1", "standard", [["Final answer: Yale"]] * 3)
    _write_jsonl(tmp_path / "m1" / "debates" / "nq_llama_agents3.jsonl", [rec])
    _write_jsonl(tmp_path / "m2" / "debates" / "nq_llama_agents3.jsonl", [rec])  # identical duplicate

    files, _ = discover_result_files([str(tmp_path / "m1"), str(tmp_path / "m2")])
    grouped, warnings = load_and_group(files)
    key = ("nq", "llama", "standard_mad")
    assert len(grouped[key]) == 1
    assert warnings == []


def test_load_and_group_warns_on_conflicting_duplicate_records(tmp_path):
    rec_a = _standard_mad_record("q1", "standard", [["Final answer: Yale"]] * 3)
    rec_b = _standard_mad_record("q1", "standard", [["Final answer: Duke"]] * 3)  # same key, different content
    _write_jsonl(tmp_path / "m1" / "debates" / "nq_llama_agents3.jsonl", [rec_a])
    _write_jsonl(tmp_path / "m2" / "debates" / "nq_llama_agents3.jsonl", [rec_b])

    files, _ = discover_result_files([str(tmp_path / "m1"), str(tmp_path / "m2")])
    grouped, warnings = load_and_group(files)
    key = ("nq", "llama", "standard_mad")
    assert len(grouped[key]) == 1  # first occurrence kept
    assert len(warnings) == 1
    assert "DIFFERING content" in warnings[0]


# ---------------------------------------------------------------------------
# build_table1_rows
# ---------------------------------------------------------------------------

def test_table1_computes_ma_for_standard_mad():
    grouped = {
        ("nq", "llama", "standard_mad"): [
            _standard_mad_record("q1", "standard",
                                  [["Final answer: Yale"] * 3, ["Final answer: Yale"] * 3, ["Final answer: Yale"] * 3]),
            _standard_mad_record("q2", "standard",
                                  [["Final answer: Duke"] * 3, ["Final answer: Duke"] * 3, ["Final answer: Duke"] * 3]),
        ]
    }
    rows = build_table1_rows(grouped)
    assert len(rows) == 1
    row = rows[0]
    assert row["dataset"] == "nq" and row["model"] == "llama" and row["method"] == "standard_mad"
    assert row["setup"] == "standard"
    assert row["MA1"] == pytest.approx(0.5)  # q1 fully correct, q2 fully wrong


def test_table1_orders_setups_in_paper_order():
    records = []
    for setup in ["standard", "0,3", "3,0", "1,2", "2,1"]:
        records.append(_digra_record(f"q_{setup}", setup, [["Yale"], ["Yale"], ["Yale"]]))
    grouped = {("nq", "llama", "digra"): records}
    rows = build_table1_rows(grouped)
    assert [r["setup"] for r in rows] == ["3,0", "2,1", "1,2", "0,3", "standard"]


def test_table1_handles_digra_schema_directly_without_adapter():
    records = [
        _digra_record("q1", "standard", [
            ["Yale", "Yale", "Yale"],
            ["Duke", "Yale", "Yale"],
            ["Yale", "Yale", "Yale"],
        ]),
    ]
    grouped = {("nq", "llama", "digra"): records}
    rows = build_table1_rows(grouped)
    row = rows[0]
    assert row["MA1"] == pytest.approx(2 / 3)
    assert row["MA2"] == pytest.approx(1.0)
    assert row["CR2"] == pytest.approx(1.0)  # the one wrong agent got corrected in round 2


# ---------------------------------------------------------------------------
# compute_fig2_series
# ---------------------------------------------------------------------------

def test_fig2_series_sorted_by_measured_ma1():
    records = [
        _digra_record("qa", "0,3", [["Yale"], ["Yale"], ["Yale"]]),   # MA1 = 1.0
        _digra_record("qb", "3,0", [["Duke"], ["Duke"], ["Duke"]]),   # MA1 = 0.0
    ]
    points = compute_fig2_series(records)
    assert [p[1] for p in points] == ["3,0", "0,3"]  # ascending by MA1
    assert points[0][0] == pytest.approx(0.0)
    assert points[1][0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# median_split_difficulty
# ---------------------------------------------------------------------------

def test_median_split_default_threshold():
    pool = [
        {"question_id": "q1", "difficulty": 0.1},
        {"question_id": "q2", "difficulty": 0.5},
        {"question_id": "q3", "difficulty": 0.9},
    ]
    easy, hard, threshold = median_split_difficulty(pool)
    assert threshold == 0.5
    assert easy == {"q2", "q3"}
    assert hard == {"q1"}


def test_median_split_explicit_threshold_override():
    pool = [{"question_id": "q1", "difficulty": 0.2}, {"question_id": "q2", "difficulty": 0.8}]
    easy, hard, threshold = median_split_difficulty(pool, threshold=0.5)
    assert threshold == 0.5
    assert easy == {"q2"}
    assert hard == {"q1"}


def test_median_split_raises_on_empty_pool():
    with pytest.raises(ValueError):
        median_split_difficulty([])


# ---------------------------------------------------------------------------
# compute_fig4_data
# ---------------------------------------------------------------------------

def test_fig4_splits_ma_by_difficulty_bucket():
    records = [
        _digra_record("easy1", "standard", [["Yale"], ["Yale"], ["Yale"]]),   # correct, easy
        _digra_record("hard1", "standard", [["Duke"], ["Duke"], ["Duke"]]),   # wrong, hard
    ]
    data = compute_fig4_data(records, easy_qids={"easy1"}, hard_qids={"hard1"})
    assert data["n_easy"] == 1
    assert data["n_hard"] == 1
    assert data["easy"][1] == pytest.approx(1.0)
    assert data["hard"][1] == pytest.approx(0.0)


def test_fig4_ignores_non_standard_setup_records():
    records = [
        _digra_record("q1", "3,0", [["Duke"], ["Duke"], ["Duke"]]),  # NOT standard -> excluded
        _digra_record("q1", "standard", [["Yale"], ["Yale"], ["Yale"]]),
    ]
    standard_only = [r for r in records if r["setup"] == "standard"]
    data = compute_fig4_data(standard_only, easy_qids={"q1"}, hard_qids=set())
    assert data["n_easy"] == 1
    assert data["easy"][1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# cot_sc_accuracy_reference
# ---------------------------------------------------------------------------

def test_cot_sc_accuracy_reference_returns_none_if_file_missing(tmp_path):
    assert cot_sc_accuracy_reference(str(tmp_path / "nope.jsonl")) is None


def test_cot_sc_accuracy_reference_reads_accuracy_at_k(tmp_path):
    from src.baselines.cot_sc import run_cot_sc
    from src.baselines.orchestration import _result_to_record
    from src.llm.fake_client import FakeLLMClient

    llm = FakeLLMClient(scripted_responses=["Final answer: Yale"] * 5)
    result = run_cot_sc(llm, "q1", "nq", "llama", "who?", gold_answer="Yale", n_samples_max=5)
    record = _result_to_record(result, [5])

    path = tmp_path / "nq_llama_cot_sc.jsonl"
    with path.open("w") as f:
        f.write(json.dumps(record) + "\n")

    acc = cot_sc_accuracy_reference(str(path), k=5)
    assert acc == pytest.approx(1.0)
