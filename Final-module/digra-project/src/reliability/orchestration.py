"""
Propagation-Control-running orchestration, factored out of
scripts/10_run_propagation_control.py so it can be tested with
FakeLLMClient (same split as every other orchestration module in this
project — src.agents.orchestration, src.baselines.orchestration,
src.baselines.mad_variant_orchestration, src.digra.orchestration).

One JSONL record per (seed, question): the full PropagationDebateResult
(agent_records + agent_reliability_records + communication_records +
rag_verification_records), via asdict() exactly like every other result
type in this project.
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
    tau_u: float,
    n_resample: int,
    theta: float,
    registry: RunRegistry,
    out_path: Union[str, Path],
    retriever=None,
    top_k_evidence: int = 3,
    alpha: float = 0.2,
    max_subset_size: Optional[int] = None,
    max_tokens: int = 300,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 50,
    logprobs_topk: int = 5,
    periodic_backup=None,
) -> dict:
    """
    Runs the propagation-control debate (setup="standard" only — see
    src.reliability.propagation_debate's module docstring; this method
    isn't run under the Woozle-effect hallucination-seeding sweep) across
    every (seed x question) in `df`, skipping already-completed runs via
    `registry`. Returns {"built": int, "skipped": int, "errors": int}.
    """
    n_built, n_skipped, n_errors = 0, 0, 0

    for seed in seeds:
        for _, row in df.iterrows():
            run_id = make_run_id(
                dataset=dataset_key, model=model_name, method=METHOD_NAME,
                seed=seed, n_agents=n_agents, lam=lam, question_id=row["question_id"],
            )
            if registry.is_done(run_id):
                n_skipped += 1
                continue

            try:
                result = run_propagation_debate(
                    llm=llm,
                    question_id=row["question_id"],
                    question=row["question"],
                    gold_answer=row["gold_answer"],
                    n_agents=n_agents,
                    n_rounds=n_rounds,
                    setup="standard",
                    lam=lam, tau_p=tau_p, tau_u=tau_u,
                    n_resample=n_resample, theta=theta,
                    gold_answer_alternatives=row.get("gold_answer_alternatives", []),
                    retriever=retriever, top_k_evidence=top_k_evidence,
                    seed=seed, alpha=alpha, max_subset_size=max_subset_size,
                    max_tokens=max_tokens, temperature=temperature,
                    top_p=top_p, top_k=top_k, logprobs_topk=logprobs_topk,
                )
            except Exception as exc:  # noqa: BLE001 — one bad question must not abort the run
                logger.error(
                    "Propagation-control debate failed for question_id=%s "
                    "(dataset=%s, model=%s, seed=%d): %s",
                    row["question_id"], dataset_key, model_name, seed, exc,
                )
                n_errors += 1
                continue

            append_jsonl(out_path, asdict(result))
            registry.mark_done(run_id)
            n_built += 1
            if periodic_backup is not None:
                periodic_backup.maybe_backup()

    return {"built": n_built, "skipped": n_skipped, "errors": n_errors}
