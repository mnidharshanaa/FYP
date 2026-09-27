"""
Checkpoint / resume infrastructure for Kaggle.

Why this exists
----------------
Kaggle sessions have hard wall-clock limits and can disconnect without
warning. A full sweep here is (3 datasets) x (2 models) x (9 methods) x
(4 seeds) x (up to 5 hallucination setups) — hundreds of independent debate
runs, each involving many LLM calls. Losing that to a disconnect at hour 8
is unacceptable, so:

  1. RunRegistry tracks which (dataset, model, method, seed, setup) units
     have already completed, persisted to disk after every single unit
     (atomic write via temp-file + rename, so a crash mid-write can't
     corrupt the registry).
  2. ResultsWriter appends result rows to a CSV incrementally and flushes
     immediately — never buffers a whole run's results in memory only to
     lose them on disconnect.

The experiment orchestrator (Module 8) checks RunRegistry.is_done(run_id)
before starting each unit, and skips it if already recorded. This makes
"just rerun the same notebook cell" a correct, cheap way to resume.
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def make_run_id(**kwargs: Any) -> str:
    """
    Build a deterministic, sorted-key run identifier from arbitrary fields, e.g.:
        make_run_id(dataset="nq", model="llama", method="digra_rag_memory",
                     seed=0, setup="1,2")
        -> "dataset=nq|method=digra_rag_memory|model=llama|seed=0|setup=1,2"
    Sorting keys makes the id independent of call-site argument order.
    """
    parts = [f"{k}={kwargs[k]}" for k in sorted(kwargs)]
    return "|".join(parts)


class RunRegistry:
    """Tracks completed run_ids so the orchestrator can skip finished work."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._completed: dict[str, dict] = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            with self.path.open("r") as f:
                data = json.load(f)
            logger.info(
                "Loaded run registry from %s (%d completed runs)",
                self.path, len(data),
            )
            return data
        logger.info("No existing run registry at %s; starting fresh", self.path)
        return {}

    def is_done(self, run_id: str) -> bool:
        return run_id in self._completed

    def mark_done(self, run_id: str, metadata: dict | None = None) -> None:
        self._completed[run_id] = metadata or {}
        self._atomic_write()
        logger.info("Marked run complete: %s", run_id)

    def _atomic_write(self) -> None:
        """Write to a temp file then os.replace — avoids a half-written
        registry file if the process is killed mid-write."""
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self.path.parent), prefix=".registry_", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self._completed, f, indent=2)
            os.replace(tmp_path, self.path)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    def __len__(self) -> int:
        return len(self._completed)


class ResultsWriter:
    """
    Append-only CSV writer for per-round result rows. Opens in append mode
    and flushes after every row — a disconnect loses at most one row, and
    the file itself is always valid up to the last successful write.
    """

    def __init__(self, path: str | Path, fieldnames: list[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = fieldnames
        file_exists = self.path.exists()

        self._file = self.path.open("a", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        if not file_exists:
            self._writer.writeheader()
            self._file.flush()

    def write_row(self, row: dict) -> None:
        missing = set(self.fieldnames) - set(row)
        extra = set(row) - set(self.fieldnames)
        if missing or extra:
            raise ValueError(
                f"Row schema mismatch. Missing={missing}, Unexpected={extra}. "
                f"Expected fieldnames={self.fieldnames}"
            )
        self._writer.writerow(row)
        self._file.flush()
        os.fsync(self._file.fileno())

    def write_rows(self, rows: Iterable[dict]) -> None:
        for row in rows:
            self.write_row(row)

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "ResultsWriter":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
