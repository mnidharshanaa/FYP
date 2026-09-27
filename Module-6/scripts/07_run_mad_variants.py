"""
scripts/07_run_mad_variants.py

Runs the sparse and random MAD baselines ("mad_sparse_half", "mad_random"
in configs/base.yaml's `baselines` list) across every dataset x
debate-model x agent-count x seed combination in the config, always under
setup="standard" (fresh, unseeded round-1 generation — see
src/baselines/mad_variants.py's module docstring for why). Does NOT
require scripts/01_build_pools.py, same reasoning as
scripts/05_run_baselines.py. Resumable.

All the actual logic lives in src/baselines/mad_variant_orchestration.py
(tested via FakeLLMClient in tests/test_mad_variant_orchestration.py);
this script only wires config + VLLMClient to it.

*** IMPORTANT: run ONE model per invocation with --model. *** Same real
Kaggle OOM failure as every other script in this project under tensor
parallelism — see scripts/02_run_debates.py's docstring.

Usage:
    python scripts/07_run_mad_variants.py --model llama
    python scripts/07_run_mad_variants.py --model mistral

    # restrict to one variant at a time, e.g. while iterating:
    python scripts/07_run_mad_variants.py --model llama --variant mad_sparse_half
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baselines.mad_variant_orchestration import run_mad_variant_for_dataset_model
from src.baselines.mad_variants import VALID_VARIANTS
from src.data.farm_loader import load_dataset
from src.llm.vllm_client import VLLMClient
from src.utils.backup import PeriodicBackup
from src.utils.checkpoint import RunRegistry
from src.utils.config import load_config
from src.utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


def main(
    config_path: str,
    overrides_path: str = None,
    model_filter: str = None,
    variant_filter: str = None,
) -> None:
    cfg = load_config(config_path, overrides=overrides_path)
    output_root = Path(cfg.project.output_root)
    setup_logging(output_root)

    registry = RunRegistry(output_root / "checkpoints" / "mad_variant_registry.json")
    out_dir = output_root / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)

    backup_cfg = cfg.backup
    periodic_backup = PeriodicBackup(
        every_n=backup_cfg.every_n_questions,
        local_dest=backup_cfg.local_dest,
        kaggle_dataset_slug=backup_cfg.kaggle_dataset_slug,
        source_dirs=[output_root / "checkpoints", output_root / "baselines"],
    )

    sparse_degree = cfg.mad_variants.sparse_degree

    variants = list(VALID_VARIANTS)
    if variant_filter is not None:
        if variant_filter not in variants:
            raise ValueError(f"--variant '{variant_filter}' must be one of {variants}")
        variants = [variant_filter]

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

            for n_agents in cfg.debate.agent_counts:
                for variant in variants:
                    out_path = out_dir / f"{dataset_key}_{model_name}_{variant}_agents{n_agents}.jsonl"
                    logger.info(
                        "Running MAD variant=%s: dataset=%s model=%s n_agents=%d -> %s",
                        variant, dataset_key, model_name, n_agents, out_path,
                    )
                    counts = run_mad_variant_for_dataset_model(
                        llm=llm,
                        variant=variant,
                        dataset_key=dataset_key,
                        model_name=model_name,
                        df=df,
                        seeds=cfg.project.seeds,
                        n_agents=n_agents,
                        n_rounds=cfg.debate.n_rounds,
                        registry=registry,
                        out_path=out_path,
                        sparse_degree=sparse_degree,
                        max_tokens=cfg.generation.max_tokens,
                        temperature=cfg.generation.temperature,
                        top_p=cfg.generation.top_p,
                        top_k=cfg.generation.top_k,
                        periodic_backup=periodic_backup,
                    )
                    logger.info(
                        "Done: variant=%s dataset=%s model=%s n_agents=%d — "
                        "built=%d skipped=%d errors=%d",
                        variant, dataset_key, model_name, n_agents,
                        counts["built"], counts["skipped"], counts["errors"],
                    )

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
    parser.add_argument(
        "--model", type=str, default=None,
        help="Restrict this run to a single model. Strongly recommended — see "
             "module docstring for the GPU OOM failure this avoids.",
    )
    parser.add_argument(
        "--variant", type=str, default=None,
        help="Restrict this run to a single variant ('mad_sparse_half' or 'mad_random'). "
             "Default: run both.",
    )
    args = parser.parse_args()
    main(args.config, args.overrides, args.model, args.variant)
