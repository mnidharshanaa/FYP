"""
scripts/03_run_digra_experiments.py

Runs DIGRA experiments (one of: digra, digra_rag, digra_rag_memory) across
every dataset x agent-count x hallucination-setup x seed combination in
the config, seeded from scripts/01_build_pools.py's pools. Resumable.

*** Same rule as scripts/01_build_pools.py and scripts/02_run_debates.py:
run ONE model per invocation with --model. *** Loading a second vLLM model
into the same process crashes with a GPU OOM error under tensor
parallelism — see those scripts' docstrings for the real failure this
caused.

*** Run --calibrate FIRST, before a full run. *** It runs a handful of
real debates (default 3), times them for real, and prints a projected
total for your full configured scope — replacing a guess with a measured
number before you commit GPU time. The calibration debates are NOT
wasted: they complete normally and count toward the resumable registry,
so a subsequent full run picks up right after them rather than redoing
them.

Usage:
    # first, on a small scope, always:
    python scripts/03_run_digra_experiments.py --model llama --variant digra --calibrate

    # then the real runs, one per variant:
    python scripts/03_run_digra_experiments.py --model llama --variant digra
    python scripts/03_run_digra_experiments.py --model llama --variant digra_rag
    python scripts/03_run_digra_experiments.py --model llama --variant digra_rag_memory
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.agents.orchestration import apply_debate_question_cap, load_pools_for_dataset_model
from src.data.farm_loader import load_dataset
from src.digra.orchestration import run_digra_experiments_for_dataset_model
from src.llm.vllm_client import VLLMClient
from src.utils.backup import PeriodicBackup
from src.utils.checkpoint import RunRegistry
from src.utils.config import load_config
from src.utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)

VALID_VARIANTS = ["digra", "digra_rag", "digra_rag_memory"]


def main(
    config_path: str,
    overrides_path: str,
    model_filter: str,
    variant: str,
    calibrate: bool,
    calibrate_n: int,
) -> None:
    if variant not in VALID_VARIANTS:
        raise ValueError(f"--variant must be one of {VALID_VARIANTS}, got '{variant}'")

    cfg = load_config(config_path, overrides=overrides_path)
    output_root = Path(cfg.project.output_root)
    setup_logging(output_root)

    registry = RunRegistry(output_root / "checkpoints" / f"digra_{variant}_registry.json")
    pools_dir = output_root / "pools"
    digra_dir = output_root / "digra"
    digra_dir.mkdir(parents=True, exist_ok=True)

    backup_cfg = cfg.backup
    periodic_backup = PeriodicBackup(
        every_n=backup_cfg.every_n_questions,
        local_dest=backup_cfg.local_dest,
        kaggle_dataset_slug=backup_cfg.kaggle_dataset_slug,
        source_dirs=[output_root / "checkpoints", output_root / "digra"],
    )

    setups = []
    for s in cfg.debate.hallucination_setups:
        setups.append("standard" if s == "standard" else tuple(s))

    debate_models = {
        name: model_cfg for name, model_cfg in cfg.models.to_dict().items()
        if isinstance(model_cfg, dict) and model_cfg.get("role") == "debate"
    }
    if model_filter is not None:
        if model_filter not in debate_models:
            raise ValueError(f"--model '{model_filter}' not found. Available: {sorted(debate_models)}")
        debate_models = {model_filter: debate_models[model_filter]}
    elif len(debate_models) > 1:
        logger.warning(
            "Running %d models in one process is NOT recommended — see module "
            "docstring. Prefer --model <name>.", len(debate_models),
        )

    digra_kwargs = dict(
        alpha=cfg.digra.alpha,
        max_subset_size=cfg.digra.max_partner_set_size,
        entropy_inflation_factor=cfg.rag.entropy_inflation_factor,
        trust_config=dict(
            initial_trust=cfg.memory.initial_trust,
            flag_penalty=cfg.memory.flag_penalty,
            consistency_reward=cfg.memory.consistency_reward,
            min_trust=cfg.memory.min_trust,
        ),
        max_tokens=cfg.generation.max_tokens,
        temperature=cfg.generation.temperature,
        top_p=cfg.generation.top_p,
        top_k=cfg.generation.top_k,
        logprobs_topk=cfg.generation.logprobs_topk,
        unchanged_rounds_threshold=cfg.digra.early_stopping.unchanged_rounds_threshold,
    )

    for model_name, model_cfg in debate_models.items():
        logger.info("Loading model '%s' for variant='%s'...", model_name, variant)
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
                dataset_key, farm_dir=cfg.data.farm_dir,
                n_questions=dataset_cfg.n_questions, seed=cfg.project.seeds[0],
            )
            df = apply_debate_question_cap(df, cfg.debate.max_questions_per_dataset)

            pools_path = pools_dir / f"{dataset_key}_{model_name}.jsonl"
            if not pools_path.exists():
                logger.error("Pools not found: %s. Run scripts/01_build_pools.py first.", pools_path)
                continue
            pools = load_pools_for_dataset_model(pools_path)

            for n_agents in cfg.debate.agent_counts:
                out_path = digra_dir / f"{dataset_key}_{model_name}_{variant}_agents{n_agents}.jsonl"
                effective_calibrate_n = calibrate_n if calibrate else None

                logger.info(
                    "Running DIGRA: dataset=%s model=%s variant=%s n_agents=%d calibrate=%s -> %s",
                    dataset_key, model_name, variant, n_agents, calibrate, out_path,
                )
                result = run_digra_experiments_for_dataset_model(
                    llm=llm, dataset_key=dataset_key, model_name=model_name, variant=variant,
                    df=df, pools=pools, setups=setups, seeds=cfg.project.seeds,
                    n_agents=n_agents, n_rounds=cfg.debate.n_rounds,
                    registry=registry, out_path=out_path, digra_kwargs=digra_kwargs,
                    periodic_backup=periodic_backup, calibrate_n=effective_calibrate_n,
                )
                logger.info("Result: %s", result)

                if calibrate and "mean_duration_seconds" in result:
                    mean_s = result["mean_duration_seconds"]
                    total_planned = len(setups) * len(df) * len(cfg.project.seeds)
                    projected_hours = (mean_s * total_planned) / 3600
                    logger.info(
                        "CALIBRATION RESULT: mean=%.1fs/debate (min=%.1fs max=%.1fs). "
                        "Projected for the full configured scope (%d debates): %.2f hours.",
                        mean_s, result["min_duration_seconds"], result["max_duration_seconds"],
                        total_planned, projected_hours,
                    )
                    return  # calibration is informational only — stop here, don't continue looping

        # Best-effort cleanup before any next model in this same process —
        # see scripts/01_build_pools.py's docstring: not reliable under
        # tensor_parallel_size > 1, --model is the actually reliable path.
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
    parser.add_argument("--overrides", type=str, default=None)
    parser.add_argument("--model", type=str, default=None, help="Restrict to one model. Strongly recommended.")
    parser.add_argument("--variant", type=str, required=True, choices=VALID_VARIANTS)
    parser.add_argument(
        "--calibrate", action="store_true",
        help="Time a handful of real debates and project a total, then stop. Run this before a full run.",
    )
    parser.add_argument("--calibrate-n", type=int, default=3)
    args = parser.parse_args()
    main(args.config, args.overrides, args.model, args.variant, args.calibrate, args.calibrate_n)
