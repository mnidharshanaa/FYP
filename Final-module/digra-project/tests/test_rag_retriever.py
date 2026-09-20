import json

import pytest

from src.rag.retriever import (
    RetrievedDocument,
    SnapshotRetriever,
    WikipediaLiveRetriever,
    load_snapshot_raw,
    save_snapshot,
)


# ---------------------------------------------------------------------------
# save_snapshot / SnapshotRetriever round-trip
# ---------------------------------------------------------------------------

def test_save_and_load_snapshot_round_trips(tmp_path):
    docs = {
        "nq_0000": [
            RetrievedDocument("wiki:Jupiter", "Jupiter", "Jupiter is the largest planet.", 1, 1.0),
            RetrievedDocument("wiki:Saturn", "Saturn", "Saturn has rings.", 2, 0.5),
        ],
    }
    path = tmp_path / "nq.json"
    save_snapshot(path, docs)

    retriever = SnapshotRetriever(path)
    assert retriever.has("nq_0000")
    assert not retriever.has("nq_9999")

    retrieved = retriever.retrieve("nq_0000", top_k=2)
    assert len(retrieved) == 2
    assert retrieved[0].document_id == "wiki:Jupiter"
    assert retrieved[0].text == "Jupiter is the largest planet."


def test_snapshot_retriever_respects_top_k():
    pass  # covered implicitly below with a 3-doc snapshot


def test_snapshot_retriever_top_k_truncates(tmp_path):
    docs = {"q1": [
        RetrievedDocument("d1", "T1", "text1", 1, 1.0),
        RetrievedDocument("d2", "T2", "text2", 2, 0.5),
        RetrievedDocument("d3", "T3", "text3", 3, 0.33),
    ]}
    path = tmp_path / "snap.json"
    save_snapshot(path, docs)
    retriever = SnapshotRetriever(path)
    assert len(retriever.retrieve("q1", top_k=1)) == 1
    assert len(retriever.retrieve("q1", top_k=10)) == 3  # never fabricates extra docs


def test_snapshot_retriever_raises_keyerror_for_missing_question():
    pass  # covered below, needs a real file


def test_snapshot_retriever_missing_question_raises_not_silently_empty(tmp_path):
    path = tmp_path / "snap.json"
    save_snapshot(path, {"q1": [RetrievedDocument("d1", "T1", "text", 1, 1.0)]})
    retriever = SnapshotRetriever(path)
    with pytest.raises(KeyError):
        retriever.retrieve("q_never_fetched", top_k=3)


def test_load_snapshot_raw_returns_empty_dict_for_missing_file(tmp_path):
    assert load_snapshot_raw(tmp_path / "does_not_exist.json") == {}


def test_load_snapshot_raw_reads_existing_file(tmp_path):
    path = tmp_path / "snap.json"
    save_snapshot(path, {"q1": [RetrievedDocument("d1", "T1", "text", 1, 1.0)]})
    raw = load_snapshot_raw(path)
    assert "q1" in raw
    assert raw["q1"][0]["document_id"] == "d1"


# ---------------------------------------------------------------------------
# WikipediaLiveRetriever — HTTP layer fully mocked, no real network call
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, json_data, status_code=200):
        self._json = json_data
        self.status_code = status_code

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeSession:
    def __init__(self, opensearch_response, summary_responses):
        self.opensearch_response = opensearch_response
        self.summary_responses = summary_responses  # dict: title -> _FakeResponse
        self.requested_urls = []

    def get(self, url, params=None, timeout=None):
        self.requested_urls.append((url, params))
        if "opensearch" in str(params):
            return self.opensearch_response
        for title, resp in self.summary_responses.items():
            if title.replace(" ", "_") in url:
                return resp
        return _FakeResponse({}, status_code=404)


def test_wikipedia_live_retriever_happy_path():
    opensearch = _FakeResponse(["largest planet", ["Jupiter", "Solar System"], [], []])
    summaries = {
        "Jupiter": _FakeResponse({"title": "Jupiter", "extract": "Jupiter is the largest planet."}),
        "Solar System": _FakeResponse({"title": "Solar System", "extract": "The Solar System has 8 planets."}),
    }
    session = _FakeSession(opensearch, summaries)
    retriever = WikipediaLiveRetriever(session=session)

    docs = retriever.retrieve("largest planet", top_k=2)
    assert len(docs) == 2
    assert docs[0].document_id == "wiki:Jupiter"
    assert docs[0].rank == 1
    assert docs[0].retrieval_score == pytest.approx(1.0)
    assert docs[1].rank == 2
    assert docs[1].retrieval_score == pytest.approx(0.5)


def test_wikipedia_live_retriever_skips_unresolvable_titles():
    opensearch = _FakeResponse(["query", ["Jupiter", "Some Disambiguation Page"], [], []])
    summaries = {
        "Jupiter": _FakeResponse({"title": "Jupiter", "extract": "text"}),
        "Some Disambiguation Page": _FakeResponse({}, status_code=404),
    }
    session = _FakeSession(opensearch, summaries)
    retriever = WikipediaLiveRetriever(session=session)

    docs = retriever.retrieve("query", top_k=2)
    assert len(docs) == 1  # the 404 page is skipped, not fabricated
    assert docs[0].document_id == "wiki:Jupiter"


def test_wikipedia_live_retriever_empty_search_returns_empty_list():
    opensearch = _FakeResponse(["nothing found", [], [], []])
    session = _FakeSession(opensearch, {})
    retriever = WikipediaLiveRetriever(session=session)
    assert retriever.retrieve("nothing found", top_k=3) == []


def test_wikipedia_live_retriever_respects_top_k():
    opensearch = _FakeResponse(["q", ["A", "B", "C"], [], []])
    summaries = {t: _FakeResponse({"title": t, "extract": f"{t} text"}) for t in ["A", "B", "C"]}
    session = _FakeSession(opensearch, summaries)
    retriever = WikipediaLiveRetriever(session=session)
    docs = retriever.retrieve("q", top_k=2)
    assert len(docs) == 2
