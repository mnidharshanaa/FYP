"""
FARM dataset loading.

Files are newline-delimited JSON (.jsonl) — one record per line, NOT a
single JSON array (verified against the real files in
pillowsofwind/llms-believe-the-earth-is-flat/src/Farm_dataset/).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import pandas as pd

from src.data.farm_schema import FarmSchemaError, normalize_record
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def load_farm_subset(path: Union[str, Path], subset: str) -> pd.DataFrame:
    """
    Load one FARM subset .jsonl file into a DataFrame of normalized records.

    Malformed lines (bad JSON, or JSON missing required fields for `subset`)
    are logged and skipped rather than crashing the whole load — a single
    corrupt line in a 491-line file shouldn't block loading the other 490,
    but every skip is logged so it's never silent.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"FARM subset file not found: {path}. "
            f"Run scripts/00_fetch_farm.py first."
        )

    records = []
    n_skipped = 0
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                normalized = normalize_record(raw, subset=subset, idx=line_no)
                records.append(normalized.to_dict())
            except (json.JSONDecodeError, FarmSchemaError) as exc:
                n_skipped += 1
                logger.warning(
                    "Skipping malformed record at %s:%d — %s", path, line_no, exc
                )

    if n_skipped > 0:
        logger.warning(
            "Loaded %s with %d skipped malformed record(s) out of %d total lines",
            path, n_skipped, line_no + 1,
        )
    else:
        logger.info("Loaded %d records from %s", len(records), path)

    if not records:
        raise ValueError(f"No valid records loaded from {path} — check the file/subset.")

    return pd.DataFrame.from_records(records)


def load_dataset(
    dataset_key: str,
    farm_dir: Union[str, Path],
    n_questions: int | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Load a config-level dataset ("nq", "boolq", "truthfulqa") into a single
    DataFrame, handling the FARM-vs-paper naming mismatch: the DIGRA paper's
    "NQ" corresponds to FARM's NQ1 + NQ2 combined.

    If `n_questions` is set, deterministically subsamples down to that many
    rows (seeded, so the same subsample is used across every method/model —
    required for the round-1 seeding to be comparable across methods).
    """
    farm_dir = Path(farm_dir)

    if dataset_key == "nq":
        nq1 = load_farm_subset(farm_dir / "NQ1.jsonl", subset="nq1")
        nq2 = load_farm_subset(farm_dir / "NQ2.jsonl", subset="nq2")
        df = pd.concat([nq1, nq2], ignore_index=True)
    elif dataset_key == "boolq":
        df = load_farm_subset(farm_dir / "Boolq.jsonl", subset="boolq")
    elif dataset_key == "truthfulqa":
        df = load_farm_subset(farm_dir / "TruthfulQA.jsonl", subset="truthfulqa")
    else:
        raise ValueError(
            f"Unknown dataset_key '{dataset_key}'. Expected one of: nq, boolq, truthfulqa"
        )

    if n_questions is not None and n_questions < len(df):
        df = df.sample(n=n_questions, random_state=seed).reset_index(drop=True)
        logger.info(
            "Subsampled %s to %d/%d questions (seed=%d)",
            dataset_key, n_questions, len(df), seed,
        )

    return df
