"""
scripts/01_build_pools.py

Builds the correct/incorrect response pool (Appendix B-3) for every
question in every dataset, for every debate-role model in the config.
Pools are written incrementally to
    {output_root}/pools/{dataset}_{model}.jsonl
one line per question, and are resumable: already-completed questions
(tracked in the RunRegistry) are skipped on re-run, so a Kaggle disconnect
loses at most the in-flight question.

This script is intentionally thin — it contains no pool-building logic of
its own; everything here is orchestration of src/data/* and src/llm/*,
per the project convention (see README "Conventions").

*** IMPORTANT: run ONE model per invocation with --model, not all models
in a single run. *** Confirmed on real Kaggle hardware (dual T4,
tensor_parallel_size=2): loading a second vLLM model into the same
Python process after finishing the first one crashes with a GPU
out-of-memory error, because vLLM's tensor-parallel worker subprocesses
do not reliably release GPU memory just from reassigning the Python
`llm` variable — the first model's weights/KV-cache/CUDA-graph memory
stays resident. The only fully reliable fix is a fresh OS process per
model, which is exactly what happens when you invoke this script twice:

    python scripts/01_build_pools.py --model llama
    python scripts/01_build_pools.py --model mistral

Each invocation is independently resumable (checkpoint registry is
shared/appended-to across both), so running them as two separate
commands loses nothing compared to one combined run — it's strictly
safer.

Usage (on Kaggle, after scripts/00_fetch_farm.py and
scripts/smoke_test_vllm.py have both been run and verified):
    python scripts/01_build_pools.py --config configs/base.yaml --model llama
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from src.data.farm_loader import load_dataset
from src.data.pool_generation import build_pool_for_question
from src.llm.vllm_client import VLLMClient
from src.utils.backup import PeriodicBackup
from src.utils.checkpoint import RunRegistry, make_run_id
from src.utils.config import load_config
from src.utils.io import append_jsonl
from src.utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


def main(config_path: str, overrides_path: str = None, model_filter: str = None) -> None:
    cfg = load_config(config_path, overrides=overrides_path)
    output_root = Path(cfg.project.output_root)
    setup_logging(output_root)

    registry = RunRegistry(output_root / "checkpoints" / "pool_build_registry.json")
    pools_dir = output_root / "pools"
    pools_dir.mkdir(parents=True, exist_ok=True)

    backup_cfg = cfg.backup
    periodic_backup = PeriodicBackup(
        every_n=backup_cfg.every_n_questions,
        local_dest=backup_cfg.local_dest,
        kaggle_dataset_slug=backup_cfg.kaggle_dataset_slug,
        source_dirs=[output_root / "checkpoints", output_root / "pools"],
    )
    logger.info(
        "Periodic backup: every_n=%d, local_dest=%s, kaggle_dataset_slug=%s",
        backup_cfg.every_n_questions, backup_cfg.local_dest, backup_cfg.kaggle_dataset_slug,
    )

    debate_models = {
        name: model_cfg for name, model_cfg in cfg.models.to_dict().items()
        if isinstance(model_cfg, dict) and model_cfg.get("role") == "debate"
    }

    if model_filter is not None:
        if model_filter not in debate_models:
            raise ValueError(
                f"--model '{model_filter}' is not a debate-role model in the config. "
                f"Available: {sorted(debate_models)}"
            )
        debate_models = {model_filter: debate_models[model_filter]}
        logger.info("Restricting this run to model: %s", model_filter)
    elif len(debate_models) > 1:
        logger.warning(
            "Running %d models (%s) in a single process. This is NOT recommended — "
            "see this script's module docstring for a real OOM failure this causes "
            "under tensor parallelism. Prefer --model <name>, one invocation per model.",
            len(debate_models), list(debate_models),
        )

    logger.info("Building pools for debate models: %s", list(debate_models))

    for model_name, model_cfg in debate_models.items():
        logger.info("Loading model '%s' (%s)...", model_name, model_cfg["hf_id"])
        llm = VLLMClient(
            model_id=model_cfg["hf_id"],
            dtype=model_cfg.get("dtype", "bfloat16"),
            max_model_len=model_cfg.get("max_model_len", 4096),
            gpu_memory_utilization=model_cfg.get("gpu_memory_utilization", 0.90),
            tensor_parallel_size=model_cfg.get("tensor_parallel_size", 1),
        )

        for dataset_key in cfg.datasets.to_dict():
            dataset_cfg = cfg.datasets[dataset_key]
            df = load_dataset(
                dataset_key,
                farm_dir=cfg.data.farm_dir,
                n_questions=dataset_cfg.n_questions,
                seed=cfg.project.seeds[0],  # fixed reference seed for question subsampling
            )
            out_path = pools_dir / f"{dataset_key}_{model_name}.jsonl"
            logger.info(
                "Building pools: dataset=%s model=%s (%d questions) -> %s",
                dataset_key, model_name, len(df), out_path,
            )

            n_skipped, n_built = 0, 0
            for _, row in df.iterrows():
                run_id = make_run_id(
                    dataset=dataset_key, model=model_name, question_id=row["question_id"],
                )
                if registry.is_done(run_id):
                    n_skipped += 1
                    continue

                pool_result = build_pool_for_question(
                    llm=llm,
                    question_id=row["question_id"],
                    question=row["question"],
                    gold_answer=row["gold_answer"],
                    alternatives=row["gold_answer_alternatives"],
                    logical_appeals=row["logical_appeals"],
                    n_all=cfg.debate.correct_pool_size,
                    n_attempts=cfg.debate.correct_pool_sampling_attempts,
                    seed=cfg.project.seeds[0],
                    max_tokens=cfg.generation.max_tokens,
                    temperature=cfg.generation.temperature,
                    top_p=cfg.generation.top_p,
                    top_k=cfg.generation.top_k,
                )

                append_jsonl(out_path, asdict(pool_result))
                registry.mark_done(run_id, metadata={"difficulty": pool_result.difficulty})
                n_built += 1
                periodic_backup.maybe_backup()

                if not pool_result.fully_satisfied:
                    logger.error(
                        "Question %s did not reach a full pool — see earlier "
                        "error log from pool_generation for details.",
                        row["question_id"],
                    )

            logger.info(
                "Done: dataset=%s model=%s — built %d, skipped %d (already done)",
                dataset_key, model_name, n_built, n_skipped,
            )

        # Best-effort cleanup before any next model in this same process.
        # NOT a reliable fix on its own under tensor_parallel_size > 1 — see
        # this script's module docstring. Kept as defense-in-depth for
        # single-GPU (tensor_parallel_size=1) setups where it tends to work,
        # but --model (one model per process) is the actually reliable path.
        import gc
        del llm
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument(
        "--overrides", type=str, default=None,
        help="Optional override YAML merged on top of --config, e.g. "
             "configs/full_scale.yaml to restore all seeds/agent-counts.",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Restrict this run to a single model (e.g. 'llama' or 'mistral'). "
             "Strongly recommended — see module docstring for why running "
             "multiple models in one process can crash with a GPU OOM error.",
    )
    args = parser.parse_args()
    main(args.config, args.overrides, args.model)
