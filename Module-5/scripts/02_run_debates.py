"""
scripts/02_run_debates.py

Runs Standard MAD (fully-connected topology) debates across every
dataset x debate-model x agent-count x hallucination-setup x seed
combination in the config, seeded from the pools built by
scripts/01_build_pools.py. Resumable — safe to re-run after a Kaggle
disconnect.

All the actual logic lives in src/agents/orchestration.py (tested via
FakeLLMClient in tests/test_orchestration.py); this script only wires
config + VLLMClient to it, per the project convention that scripts/
contain no logic of their own.

*** IMPORTANT: run ONE model per invocation with --model, not all models
in a single run. *** Same real Kaggle failure as scripts/01_build_pools.py
(see that script's docstring) — loading a second vLLM model into the same
process after finishing the first crashes with a GPU OOM error under
tensor parallelism. Run:

    python scripts/02_run_debates.py --model llama
    python scripts/02_run_debates.py --model mistral

Usage (after scripts/00_fetch_farm.py, scripts/smoke_test_vllm.py, and
scripts/01_build_pools.py have all completed):
    python scripts/02_run_debates.py --config configs/base.yaml --model llama
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.agents.orchestration import (
    apply_debate_question_cap,
    load_pools_for_dataset_model,
    parse_setup,
    run_debates_for_dataset_model,
)
from src.data.farm_loader import load_dataset
from src.llm.vllm_client import VLLMClient
from src.utils.backup import PeriodicBackup
from src.utils.checkpoint import RunRegistry
from src.utils.config import load_config
from src.utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


def main(config_path: str, overrides_path: str = None, model_filter: str = None) -> None:
    cfg = load_config(config_path, overrides=overrides_path)
    output_root = Path(cfg.project.output_root)
    setup_logging(output_root)

    registry = RunRegistry(output_root / "checkpoints" / "debate_registry.json")
    pools_dir = output_root / "pools"
    debates_dir = output_root / "debates"
    debates_dir.mkdir(parents=True, exist_ok=True)

    backup_cfg = cfg.backup
    periodic_backup = PeriodicBackup(
        every_n=backup_cfg.every_n_questions,
        local_dest=backup_cfg.local_dest,
        kaggle_dataset_slug=backup_cfg.kaggle_dataset_slug,
        source_dirs=[output_root / "checkpoints", output_root / "debates"],
    )

    setups = [parse_setup(s) for s in cfg.debate.hallucination_setups]

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
                seed=cfg.project.seeds[0],
            )
            df = apply_debate_question_cap(df, cfg.debate.max_questions_per_dataset)

            pools_path = pools_dir / f"{dataset_key}_{model_name}.jsonl"
            if not pools_path.exists():
                logger.error(
                    "Pools file not found: %s. Run scripts/01_build_pools.py "
                    "first. Skipping dataset=%s model=%s entirely.",
                    pools_path, dataset_key, model_name,
                )
                continue
            pools = load_pools_for_dataset_model(pools_path)

            for n_agents in cfg.debate.agent_counts:
                out_path = debates_dir / f"{dataset_key}_{model_name}_agents{n_agents}.jsonl"
                logger.info(
                    "Running debates: dataset=%s model=%s n_agents=%d -> %s",
                    dataset_key, model_name, n_agents, out_path,
                )
                counts = run_debates_for_dataset_model(
                    llm=llm,
                    dataset_key=dataset_key,
                    model_name=model_name,
                    df=df,
                    pools=pools,
                    setups=setups,
                    seeds=cfg.project.seeds,
                    n_agents=n_agents,
                    n_rounds=cfg.debate.n_rounds,
                    registry=registry,
                    out_path=out_path,
                    max_tokens=cfg.generation.max_tokens,
                    temperature=cfg.generation.temperature,
                    top_p=cfg.generation.top_p,
                    top_k=cfg.generation.top_k,
                    periodic_backup=periodic_backup,
                )
                logger.info(
                    "Done: dataset=%s model=%s n_agents=%d — built=%d skipped=%d errors=%d",
                    dataset_key, model_name, n_agents,
                    counts["built"], counts["skipped"], counts["errors"],
                )

        # Best-effort cleanup before any next model in this same process —
        # see module docstring: not reliable under tensor_parallel_size > 1,
        # --model is the actually reliable path.
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
