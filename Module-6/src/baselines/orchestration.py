"""
CoT-SC-running orchestration, factored out of scripts/05_run_baselines.py
so it can be tested with FakeLLMClient (same split as
src/agents/orchestration.py / scripts/02_run_debates.py).

One JSONL record per (dataset, model, seed, question): the full set of
n_samples_max raw samples plus a precomputed self-consistency vote at
every k in `sc_ks`, so scripts/06_aggregate_baselines.py never needs to
re-run a model to answer "what if we'd only sampled 3?" — see
src/baselines/cot_sc.py's module docstring for why this is generated
once at max N rather than once per k.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Union

import pandas as pd

from src.baselines.cot_sc import majority_vote, run_cot_sc
from src.llm.client import LLMClient
from src.utils.checkpoint import RunRegistry, make_run_id
from src.utils.io import append_jsonl
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

METHOD_NAME = "cot_sc"


def _result_to_record(result, sc_ks: list) -> dict:
    record = asdict(result)
    record["votes"] = {
        str(k): asdict(majority_vote(result, k)) for k in sc_ks if k <= result.n_samples_requested
    }
    return record


def run_baselines_for_dataset_model(
    llm: LLMClient,
    dataset_key: str,
    model_name: str,
    df: pd.DataFrame,
    seeds: list,
    n_samples_max: int,
    sc_ks: list,
    registry: RunRegistry,
    out_path: Union[str, Path],
    max_tokens: int = 300,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 50,
    logprobs_topk=None,
    periodic_backup=None,
) -> dict:
    """
    Run CoT-SC (n_samples_max fresh samples per question, self-consistency
    vote precomputed at every k in sc_ks) for every (seed x question) in
    `df`, skipping already-completed runs via `registry`.

    Returns {"built": int, "skipped": int, "errors": int}.
    """
    n_built, n_skipped, n_errors = 0, 0, 0

    for seed in seeds:
        for _, row in df.iterrows():
            run_id = make_run_id(
                dataset=dataset_key, model=model_name, method=METHOD_NAME,
                seed=seed, question_id=row["question_id"],
            )
            if registry.is_done(run_id):
                n_skipped += 1
                continue

            try:
                start = time.perf_counter()
                result = run_cot_sc(
                    llm=llm,
                    question_id=row["question_id"],
                    dataset=dataset_key,
                    model_name=model_name,
                    question=row["question"],
                    gold_answer=row["gold_answer"],
                    gold_answer_alternatives=row.get("gold_answer_alternatives", []),
                    n_samples_max=n_samples_max,
                    seed=seed,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    logprobs_topk=logprobs_topk,
                )
                duration = time.perf_counter() - start
            except Exception as exc:  # noqa: BLE001 — one bad question must not abort the run
                logger.error(
                    "CoT-SC failed for question_id=%s (dataset=%s, model=%s, seed=%d): %s",
                    row["question_id"], dataset_key, model_name, seed, exc,
                )
                n_errors += 1
                continue

            append_jsonl(out_path, _result_to_record(result, sc_ks))
            registry.mark_done(run_id, metadata={"duration_seconds": duration})
            n_built += 1
            if periodic_backup is not None:
                periodic_backup.maybe_backup()

    return {"built": n_built, "skipped": n_skipped, "errors": n_errors}
