"""
MAD-variant-running orchestration, factored out of
scripts/07_run_mad_variants.py so it can be tested with FakeLLMClient
(same split as src/agents/orchestration.py / scripts/02_run_debates.py).

One JSONL record per (variant, seed, question), in exactly the same
DebateResult shape Standard MAD writes (src/agents/debate.py) — so
src/metrics/standard_mad_adapter.py and everything built on top of it
(propagation_metrics, scripts/04_generate_report.py) work on these files
completely unmodified; only the filename tags which variant produced them.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from src.baselines.mad_variants import run_mad_variant_debate
from src.llm.client import LLMClient
from src.utils.checkpoint import RunRegistry, make_run_id
from src.utils.io import append_jsonl
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def run_mad_variant_for_dataset_model(
    llm: LLMClient,
    variant: str,
    dataset_key: str,
    model_name: str,
    df: pd.DataFrame,
    seeds: list,
    n_agents: int,
    n_rounds: int,
    registry: RunRegistry,
    out_path: Union[str, Path],
    sparse_degree: Optional[float] = None,
    max_tokens: int = 300,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 50,
    periodic_backup=None,
) -> dict:
    """
    Run one MAD variant ("mad_sparse_half" or "mad_random") across every
    (seed x question) in `df`, always under setup="standard" (see
    src/baselines/mad_variants.py's module docstring for why), skipping
    already-completed runs via `registry`.

    Returns {"built": int, "skipped": int, "errors": int}.
    """
    n_built, n_skipped, n_errors = 0, 0, 0

    for seed in seeds:
        for _, row in df.iterrows():
            run_id = make_run_id(
                dataset=dataset_key, model=model_name, method=variant,
                seed=seed, n_agents=n_agents, question_id=row["question_id"],
            )
            if registry.is_done(run_id):
                n_skipped += 1
                continue

            try:
                result = run_mad_variant_debate(
                    llm=llm,
                    variant=variant,
                    question_id=row["question_id"],
                    question=row["question"],
                    gold_answer=row["gold_answer"],
                    gold_answer_alternatives=row.get("gold_answer_alternatives", []),
                    n_agents=n_agents,
                    n_rounds=n_rounds,
                    seed=seed,
                    sparse_degree=sparse_degree,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
            except Exception as exc:  # noqa: BLE001 — one bad question must not abort the run
                logger.error(
                    "MAD variant=%s failed for question_id=%s (dataset=%s, model=%s, seed=%d): %s",
                    variant, row["question_id"], dataset_key, model_name, seed, exc,
                )
                n_errors += 1
                continue

            append_jsonl(out_path, asdict(result))
            registry.mark_done(run_id)
            n_built += 1
            if periodic_backup is not None:
                periodic_backup.maybe_backup()

    return {"built": n_built, "skipped": n_skipped, "errors": n_errors}
