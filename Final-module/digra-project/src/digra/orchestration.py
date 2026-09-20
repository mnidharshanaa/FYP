"""
DIGRA experiment orchestration, factored out of scripts/03_run_digra_experiments.py
the same way src/agents/orchestration.py is factored out of scripts/02_run_debates.py —
testable via FakeLLMClient, unlike the script itself which constructs a real VLLMClient.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Union

import pandas as pd

from src.digra.digra_debate import run_digra_debate
from src.llm.client import LLMClient
from src.utils.checkpoint import RunRegistry, make_run_id
from src.utils.io import append_jsonl
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def run_digra_experiments_for_dataset_model(
    llm: LLMClient,
    dataset_key: str,
    model_name: str,
    variant: str,
    df: pd.DataFrame,
    pools: dict,
    setups: list,
    seeds: list,
    n_agents: int,
    n_rounds: int,
    registry: RunRegistry,
    out_path: Union[str, Path],
    digra_kwargs: dict = None,
    periodic_backup=None,
    calibrate_n: int = None,
) -> dict:
    """
    Runs every (setup x seed x question) combination for one (dataset,
    model, variant) slice, same resumability/skip-mismatch conventions as
    src/agents/orchestration.py's run_debates_for_dataset_model.

    calibrate_n: if set, stops after this many REAL debates have been run
    (not a separate dry-run mode — these debates complete normally and
    count toward the registry, nothing is wasted), and returns timing
    stats instead of/alongside the usual counts, so the caller can project
    a realistic total time before committing to the full configured scope.
    """
    digra_kwargs = digra_kwargs or {}
    n_built, n_skipped, n_errors = 0, 0, 0
    debate_durations = []

    for setup in setups:
        label = "standard" if setup == "standard" else f"{setup[0]},{setup[1]}"
        if setup != "standard" and sum(setup) != n_agents:
            logger.info(
                "Skipping setup=%s for n_agents=%d (counts don't match).", label, n_agents,
            )
            continue

        for seed in seeds:
            for _, row in df.iterrows():
                if calibrate_n is not None and len(debate_durations) >= calibrate_n:
                    return _summarize(n_built, n_skipped, n_errors, debate_durations, calibrated=True)

                run_id = make_run_id(
                    dataset=dataset_key, model=model_name, variant=variant, setup=label,
                    seed=seed, n_agents=n_agents, question_id=row["question_id"],
                )
                if registry.is_done(run_id):
                    n_skipped += 1
                    continue

                pool = pools.get(row["question_id"])
                if pool is None and setup != "standard":
                    logger.error(
                        "No pool for question_id=%s. Run scripts/01_build_pools.py first. Skipping.",
                        row["question_id"],
                    )
                    n_errors += 1
                    continue

                start = time.monotonic()
                try:
                    result = run_digra_debate(
                        llm=llm,
                        question_id=row["question_id"],
                        question=row["question"],
                        gold_answer=row["gold_answer"],
                        gold_answer_alternatives=row.get("gold_answer_alternatives", []),
                        n_agents=n_agents,
                        n_rounds=n_rounds,
                        setup=setup,
                        variant=variant,
                        correct_pool=(pool or {}).get("correct_texts", []),
                        incorrect_pool=(pool or {}).get("incorrect_texts", []),
                        source_passage=row.get("source"),
                        seed=seed,
                        **digra_kwargs,
                    )
                except ValueError as exc:
                    logger.error(
                        "Debate failed for question_id=%s: %s. Skipping.", row["question_id"], exc,
                    )
                    n_errors += 1
                    continue
                duration = time.monotonic() - start
                debate_durations.append(duration)

                append_jsonl(out_path, asdict(result))
                registry.mark_done(run_id, metadata={"duration_seconds": duration})
                n_built += 1
                if periodic_backup is not None:
                    periodic_backup.maybe_backup()

    return _summarize(n_built, n_skipped, n_errors, debate_durations, calibrated=False)


def _summarize(n_built, n_skipped, n_errors, durations, calibrated: bool) -> dict:
    summary = {"built": n_built, "skipped": n_skipped, "errors": n_errors, "calibrated": calibrated}
    if durations:
        summary["mean_duration_seconds"] = sum(durations) / len(durations)
        summary["min_duration_seconds"] = min(durations)
        summary["max_duration_seconds"] = max(durations)
    return summary
