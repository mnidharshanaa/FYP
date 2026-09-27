import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.rag.retriever import RetrievedDocument, load_snapshot_raw


def _load_script_module():
    path = Path(__file__).resolve().parent.parent / "scripts" / "09_build_evidence_snapshot.py"
    spec = importlib.util.spec_from_file_location("build_evidence_snapshot", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def mod():
    return _load_script_module()


class _FakeRetriever:
    def __init__(self, fail_on=None):
        self.calls = []
        self.fail_on = fail_on or set()

    def retrieve(self, query, top_k):
        self.calls.append(query)
        if query in self.fail_on:
            raise RuntimeError("simulated network failure")
        return [RetrievedDocument("d1", "T", f"evidence for {query}", 1, 1.0)]


def _tiny_df():
    return pd.DataFrame([
        {"question_id": "q1", "question": "Who won?"},
        {"question_id": "q2", "question": "Who lost?"},
    ])


def test_first_run_fetches_every_question(tmp_path):
    mod = _load_script_module()
    fake = _FakeRetriever()
    out_path = tmp_path / "nq.json"
    counts = mod.build_snapshot_for_dataset("nq", _tiny_df(), out_path, top_k=1,
                                             request_delay_seconds=0, retriever=fake)
    assert counts == {"fetched": 2, "skipped": 0, "errors": 0, "total": 2}
    assert fake.calls == ["Who won?", "Who lost?"]


def test_second_run_skips_already_fetched_questions(tmp_path):
    mod = _load_script_module()
    out_path = tmp_path / "nq.json"
    mod.build_snapshot_for_dataset("nq", _tiny_df(), out_path, top_k=1,
                                    request_delay_seconds=0, retriever=_FakeRetriever())

    fake2 = _FakeRetriever()
    counts = mod.build_snapshot_for_dataset("nq", _tiny_df(), out_path, top_k=1,
                                             request_delay_seconds=0, retriever=fake2)
    assert counts == {"fetched": 0, "skipped": 2, "errors": 0, "total": 2}
    assert fake2.calls == []  # no network calls at all for already-fetched questions


def test_a_failed_question_does_not_lose_earlier_progress(tmp_path):
    mod = _load_script_module()
    out_path = tmp_path / "nq.json"
    fake = _FakeRetriever(fail_on={"Who lost?"})
    counts = mod.build_snapshot_for_dataset("nq", _tiny_df(), out_path, top_k=1,
                                             request_delay_seconds=0, retriever=fake)
    assert counts == {"fetched": 1, "skipped": 0, "errors": 1, "total": 1}

    raw = load_snapshot_raw(out_path)
    assert "q1" in raw
    assert "q2" not in raw  # the failed one is absent, not silently faked


def test_retrying_after_a_failure_only_refetches_the_missing_question(tmp_path):
    mod = _load_script_module()
    out_path = tmp_path / "nq.json"
    mod.build_snapshot_for_dataset("nq", _tiny_df(), out_path, top_k=1,
                                    request_delay_seconds=0, retriever=_FakeRetriever(fail_on={"Who lost?"}))

    fake_retry = _FakeRetriever()  # this time it succeeds
    counts = mod.build_snapshot_for_dataset("nq", _tiny_df(), out_path, top_k=1,
                                             request_delay_seconds=0, retriever=fake_retry)
    assert counts == {"fetched": 1, "skipped": 1, "errors": 0, "total": 2}
    assert fake_retry.calls == ["Who lost?"]  # only the previously-failed question is refetched
