"""
Real (non-oracle) evidence retrieval for the Propagation Control Module's
conditional RAG stage. Deliberately separate from src/rag/evidence.py's
existing oracle-passage proxy — that module stays exactly as-is; it
already has run data under the "digra_rag"/"digra_rag_memory" methods and
this file does not touch it.

Two implementations of the same interface:
  - SnapshotRetriever: reads a frozen local JSON file. This is what the
    actual GPU experiment calls — fully offline, byte-identical on every
    rerun regardless of Wikipedia's live state tomorrow.
  - WikipediaLiveRetriever: hits Wikipedia's public REST API. ONLY used by
    scripts/09_build_evidence_snapshot.py, a separate, one-time, non-GPU,
    internet-requiring step. Never import this into anything that runs
    during the main experiment.

Corpus-agnostic by design: RetrievedDocument and the retriever interface
don't know or care which implementation produced them, so a different
evidence source can be swapped in later (a local corpus, a different
API) without touching the verification or orchestration code that
consumes retrieved documents.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Union


@dataclass(frozen=True)
class RetrievedDocument:
    document_id: str
    title: str
    text: str
    rank: int
    retrieval_score: float


class SnapshotRetriever:
    """
    Offline retriever reading a frozen snapshot built by
    scripts/09_build_evidence_snapshot.py. The ONLY retriever the main
    experiment should use — zero network calls at experiment time.

    Snapshot format: {question_id: [{"document_id","title","text","rank",
    "retrieval_score"}, ...]}, keyed by question_id (not raw query text) —
    question_id is the stable join key with the rest of the pipeline;
    the query text actually used to fetch each question's documents was
    the question text itself, decided once at fetch time.
    """

    def __init__(self, snapshot_path: Union[str, Path]):
        self.snapshot_path = Path(snapshot_path)
        with self.snapshot_path.open() as f:
            self._data = json.load(f)

    def has(self, question_id: str) -> bool:
        return question_id in self._data

    def retrieve(self, question_id: str, top_k: int) -> list:
        """
        Raises KeyError if `question_id` isn't in the snapshot. This is
        intentional: a missing question means the one-time fetch step
        never covered it, and the caller must decide explicitly how to
        handle "no evidence available" (see src/rag/verification.py's
        explicit rejection of empty document lists) — silently returning
        an empty list here would look identical to "searched and found
        nothing," which is a different, genuine case.
        """
        if question_id not in self._data:
            raise KeyError(
                f"question_id={question_id!r} not found in evidence snapshot "
                f"{self.snapshot_path}. Re-run scripts/09_build_evidence_snapshot.py "
                f"to include it, or handle missing-evidence explicitly at the call site."
            )
        docs = self._data[question_id][:top_k]
        return [RetrievedDocument(**d) for d in docs]


def save_snapshot(path: Union[str, Path], data: dict) -> None:
    """
    `data`: {question_id: [RetrievedDocument, ...]}. Plain JSON (asdict on
    every document), never pickled, so the frozen file stays human-
    inspectable and independent of any Python version/class-layout change.
    """
    serializable = {qid: [asdict(d) for d in docs] for qid, docs in data.items()}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(serializable, f, indent=2)


def load_snapshot_raw(path: Union[str, Path]) -> dict:
    """Plain dict load (question_id -> list of raw dicts), used by the
    snapshot-build script to check what's already fetched for resumability,
    without needing a full SnapshotRetriever (which requires the file to
    already fully exist)."""
    path = Path(path)
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


class WikipediaLiveRetriever:
    """
    Live Wikipedia retrieval — ONLY for scripts/09_build_evidence_snapshot.py.

    Two-step per query: opensearch (candidate titles) -> page summary
    (actual extract text) per candidate, both against Wikipedia's public
    REST API, no API key required. `session` is injectable purely so
    tests can substitute a fake HTTP layer without any network access —
    production code should just use the default.
    """

    OPENSEARCH_URL = "https://en.wikipedia.org/w/api.php"
    SUMMARY_URL_TEMPLATE = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"

    def __init__(self, session=None, timeout: float = 10.0):
        if session is None:
            import requests  # local import: only needed for the real live
                              # path, never during GPU-experiment imports
            session = requests.Session()
        self.session = session
        self.timeout = timeout

    def _search_titles(self, query: str, limit: int) -> list:
        resp = self.session.get(
            self.OPENSEARCH_URL,
            params={"action": "opensearch", "search": query, "limit": limit, "format": "json"},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data[1] if len(data) > 1 else []

    def retrieve(self, query: str, top_k: int) -> list:
        titles = self._search_titles(query, limit=top_k)
        documents = []
        for rank, title in enumerate(titles[:top_k], start=1):
            resp = self.session.get(
                self.SUMMARY_URL_TEMPLATE.format(title=title.replace(" ", "_")),
                timeout=self.timeout,
            )
            if resp.status_code != 200:
                continue  # disambiguation pages / redirects the summary endpoint can't resolve
            data = resp.json()
            documents.append(RetrievedDocument(
                document_id=f"wiki:{title}",
                title=data.get("title", title),
                text=data.get("extract", ""),
                rank=rank,
                retrieval_score=1.0 / rank,  # Wikipedia exposes relevance ORDER, not a numeric score
            ))
        return documents
