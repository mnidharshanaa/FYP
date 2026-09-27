"""
Propagation-Control-running orchestration, factored out of
scripts/10_run_propagation_control.py so it can be tested with
FakeLLMClient (same split as every other orchestration module in this
project — src.agents.orchestration, src.baselines.orchestration,
src.baselines.mad_variant_orchestration, src.digra.orchestration).

Runs a SETUP SWEEP, mirroring src.digra.orchestration's pattern exactly:
"standard" (genuine, unseeded generation) PLUS, per the frozen decision to
ADD rather than replace, the controlled-correctness conditions (e.g.
(1,2) = "1 agent seeded wrong, 2 seeded correct") — the direct propagation
experiment Section 46 of the spec calls for. Controlled setups need
per-question pools (correct_texts/incorrect_texts), same `pools` dict
shape as src.digra.orchestration already uses — build them with
scripts/01_build_pools.py before requesting anything other than
"standard" here.

One JSONL record per (setup, seed, question): the full
PropagationDebateResult (agent_records + agent_reliability_records +
communication_records + rag_verification_records), via asdict() exactly
like every other result type in this project.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from src.llm.client import LLMClient
from src.reliability.propagation_debate import run_propagation_debate
from src.utils.checkpoint import RunRegistry, make_run_id
from src.utils.io import append_jsonl
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

METHOD_NAME = "propagation_control"


def run_propagation_control_for_dataset_model(
    llm: LLMClient,
    dataset_key: str,
    model_name: str,
    df: pd.DataFrame,
    seeds: list,
    n_agents: int,
    n_rounds: int,
    lam: float,
    tau_p: float,
    tau_l: float,
    n_resample: int,
    theta: float,
    registry: RunRegistry,
    out_path: Union[str, Path],
    setups: Optional[list] = None,
    pools: Optional[dict] = None,
    retriever=None,
    top_k_evidence: int = 3,
    alpha: float = 0.2,
    max_subset_size: Optional[int] = None,
    max_tokens: int = 300,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 50,
    logprobs_topk: int = 5,
    bypass_reliability_gate: bool = False,
    periodic_backup=None,
) -> dict:
    """
    Runs every (setup x seed x question) combination for this (dataset,
    model), skipping already-completed runs via `registry`. `setups`
    defaults to `["standard"]` if not given — pass e.g.
    `["standard", (1, 2), (2, 1)]` etc. to add controlled-correctness
    conditions (see module docstring). Any non-"standard" setup requires
    `pools` (a dict keyed by question_id, `{"correct_texts": [...],
    "incorrect_texts": [...]}`, same shape scripts/01_build_pools.py
    produces) — a missing pool for a requested controlled setup is logged
    as an error and that (setup, question) combination is skipped, not
    silently run as if it were standard.

    `bypass_reliability_gate`: Condition B ("DIGRA routing only," see
    src.reliability.propagation_debate's module docstring) — passed
    straight through.

    Returns {"built": int, "skipped": int, "errors": int}.
    """
    setups = setups or ["standard"]
    pools = pools or {}
    n_built, n_skipped, n_errors = 0, 0, 0

    for setup in setups:
        label = "standard" if setup == "standard" else f"{setup[0]},{setup[1]}"
        if setup != "standard" and sum(setup) != n_agents:
            logger.info("Skipping setup=%s for n_agents=%d (counts don't match).", label, n_agents)
            continue

        for seed in seeds:
            for _, row in df.iterrows():
                run_id = make_run_id(
                    dataset=dataset_key, model=model_name, method=METHOD_NAME, setup=label,
                    seed=seed, n_agents=n_agents, lam=lam,
                    bypass_gate=bypass_reliability_gate, question_id=row["question_id"],
                )
                if registry.is_done(run_id):
                    n_skipped += 1
                    continue

                pool = pools.get(row["question_id"])
                if pool is None and setup != "standard":
                    logger.error(
                        "No pool for question_id=%s (setup=%s). Run scripts/01_build_pools.py "
                        "first. Skipping.", row["question_id"], label,
                    )
                    n_errors += 1
                    continue

                try:
                    result = run_propagation_debate(
                        llm=llm,
                        question_id=row["question_id"],
                        question=row["question"],
                        gold_answer=row["gold_answer"],
                        n_agents=n_agents,
                        n_rounds=n_rounds,
                        setup=setup,
                        lam=lam, tau_p=tau_p, tau_l=tau_l,
                        n_resample=n_resample, theta=theta,
                        gold_answer_alternatives=row.get("gold_answer_alternatives", []),
                        correct_pool=(pool or {}).get("correct_texts", []),
                        incorrect_pool=(pool or {}).get("incorrect_texts", []),
                        retriever=retriever, top_k_evidence=top_k_evidence,
                        seed=seed, alpha=alpha, max_subset_size=max_subset_size,
                        max_tokens=max_tokens, temperature=temperature,
                        top_p=top_p, top_k=top_k, logprobs_topk=logprobs_topk,
                        bypass_reliability_gate=bypass_reliability_gate,
                    )
                except Exception as exc:  # noqa: BLE001 — one bad question must not abort the run
                    logger.error(
                        "Propagation-control debate failed for question_id=%s "
                        "(dataset=%s, model=%s, setup=%s, seed=%d): %s",
                        row["question_id"], dataset_key, model_name, label, seed, exc,
                    )
                    n_errors += 1
                    continue

                append_jsonl(out_path, asdict(result))
                registry.mark_done(run_id)
                n_built += 1
                if periodic_backup is not None:
                    periodic_backup.maybe_backup()

    return {"built": n_built, "skipped": n_skipped, "errors": n_errors}
