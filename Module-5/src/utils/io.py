"""
JSONL append/read helpers.

Why this exists separately from ResultsWriter (checkpoint.py)
----------------------------------------------------------------
ResultsWriter is CSV-based and expects a fixed flat schema — a good fit
for per-round metric rows (MA/MR/IMR/CR are all scalars). Pool results
contain nested lists (correct_texts, incorrect_texts), which don't fit a
flat CSV row well. JSONL (one JSON object per line) handles nested
structure naturally while keeping the same resume-safety property:
appending one line at a time, flushed immediately, so a Kaggle disconnect
loses at most the one in-flight record.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Union


def append_jsonl(path: Union[str, Path], obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")
        f.flush()


def read_jsonl(path: Union[str, Path]) -> Iterator[dict]:
    path = Path(path)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_jsonl_ids(path: Union[str, Path], id_field: str) -> set:
    """Convenience for resume checks: the set of `id_field` values already
    present in a JSONL file, without loading full records into memory."""
    return {record[id_field] for record in read_jsonl(path)}
