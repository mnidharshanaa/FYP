"""
Debate-running orchestration, factored out of scripts/02_run_debates.py so
it can be tested with FakeLLMClient (unlike the script itself, which
constructs a real VLLMClient and therefore can't be unit-tested here).
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Union

import pandas as pd

from src.agents.debate import run_debate
from src.agents.round_seeding import InsufficientPoolError
from src.llm.client import LLMClient
from src.utils.checkpoint import RunRegistry, make_run_id
from src.utils.io import append_jsonl, read_jsonl
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def apply_debate_question_cap(df: pd.DataFrame, max_questions) -> pd.DataFrame:
    """
    Truncate `df` to its first `max_questions` rows (deterministic — always
    the same prefix given the same df), or return it unchanged if
    max_questions is None or >= len(df).

    Deliberately a prefix slice, NOT a fresh random re-sample: `df` here is
    always produced by the same load_dataset(..., n_questions=X, seed=Y)
    call used when the pools for these questions were built (see
    scripts/01_build_pools.py and scripts/02_run_debates.py), so a prefix
    slice is guaranteed to only select questions that already have pools
    built — re-sampling with a smaller n_questions instead would silently
    pick a different subset and orphan already-completed pool-building work.
    """
    if max_questions is None or max_questions >= len(df):
        return df
    return df.iloc[:max_questions].reset_index(drop=True)


def parse_setup(raw_setup):
    """Config setups are either the string "standard" or a 2-element list
    [n_incorrect, n_correct] (YAML lists deserialize as plain Python lists,
    not Config-wrapped, since their elements are scalars — see
    src/utils/config.py's `_wrap`). Normalize the list form to a tuple."""
    if raw_setup == "standard":
        return "standard"
    return tuple(raw_setup)


def setup_label(setup) -> str:
    return "standard" if setup == "standard" else f"{setup[0]},{setup[1]}"


def load_pools_for_dataset_model(pools_path: Union[str, Path]) -> dict:
    """Load a pools JSONL (from scripts/01_build_pools.py) into a
    question_id -> {"correct_texts": [...], "incorrect_texts": [...]} map."""
    pools = {}
    for record in read_jsonl(pools_path):
        pools[record["question_id"]] = {
            "correct_texts": record["correct_texts"],
            "incorrect_texts": record["incorrect_texts"],
        }
    return pools


def run_debates_for_dataset_model(
    llm: LLMClient,
    dataset_key: str,
    model_name: str,
    df: pd.DataFrame,
    pools: dict,
    setups: list,
    seeds: list,
    n_agents: int,
    n_rounds: int,
    registry: RunRegistry,
    out_path: Union[str, Path],
    max_tokens: int = 300,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 50,
    periodic_backup=None,   # optional src.utils.backup.PeriodicBackup instance
) -> dict:
    """
    Run every (setup x seed x question) combination for one (dataset, model,
    n_agents) slice, skipping already-completed runs (registry) and setups
    whose (n_incorrect, n_correct) doesn't sum to n_agents (e.g. the
    default hallucination_setups in configs/base.yaml are defined for
    n_agents=3 — a 5-agent sweep will legitimately skip those, this is not
    an error).

    Returns a dict of counts: {"built": int, "skipped": int, "errors": int}
    for the caller to log/aggregate.
    """
    n_built, n_skipped, n_errors = 0, 0, 0

    for setup in setups:
        label = setup_label(setup)
        if setup != "standard" and sum(setup) != n_agents:
            logger.info(
                "Skipping setup=%s for n_agents=%d (counts don't match); "
                "this setup was defined for a different agent count.",
                label, n_agents,
            )
            continue

        for seed in seeds:
            for _, row in df.iterrows():
                run_id = make_run_id(
                    dataset=dataset_key, model=model_name, setup=label,
                    seed=seed, n_agents=n_agents, question_id=row["question_id"],
                )
                if registry.is_done(run_id):
                    n_skipped += 1
                    continue

                pool = pools.get(row["question_id"])
                if pool is None and setup != "standard":
                    logger.error(
                        "No pool found for question_id=%s (dataset=%s, model=%s). "
                        "Run scripts/01_build_pools.py first. Skipping this run.",
                        row["question_id"], dataset_key, model_name,
                    )
                    n_errors += 1
                    continue

                try:
                    result = run_debate(
                        llm=llm,
                        question_id=row["question_id"],
                        question=row["question"],
                        gold_answer=row["gold_answer"],
                        gold_answer_alternatives=row.get("gold_answer_alternatives", []),
                        n_agents=n_agents,
                        n_rounds=n_rounds,
                        setup=setup,
                        correct_pool=(pool or {}).get("correct_texts", []),
                        incorrect_pool=(pool or {}).get("incorrect_texts", []),
                        seed=seed,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                    )
                except InsufficientPoolError as exc:
                    logger.error(
                        "Insufficient pool for question_id=%s setup=%s: %s. Skipping.",
                        row["question_id"], label, exc,
                    )
                    n_errors += 1
                    continue

                append_jsonl(out_path, asdict(result))
                registry.mark_done(run_id)
                n_built += 1
                if periodic_backup is not None:
                    periodic_backup.maybe_backup()

    return {"built": n_built, "skipped": n_skipped, "errors": n_errors}
