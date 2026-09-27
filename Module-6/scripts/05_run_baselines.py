"""
scripts/05_run_baselines.py

Module 1: runs the CoT / CoT-SC baseline (src/baselines/cot_sc.py) across
every dataset x debate-model x seed combination in the config. Fresh,
independent sampling from a neutral prompt — deliberately does NOT touch
the response pools from scripts/01_build_pools.py (see
src/baselines/cot_sc.py's module docstring for why reusing them would
bias this baseline's accuracy upward). Resumable — safe to re-run after a
Kaggle disconnect.

All the actual logic lives in src/baselines/orchestration.py (tested via
FakeLLMClient in tests/test_baseline_orchestration.py); this script only
wires config + VLLMClient to it, per the project convention that scripts/
contain no logic of their own.

*** IMPORTANT: run ONE model per invocation with --model, not all models
in a single run. *** Same real Kaggle failure as scripts/01_build_pools.py
and scripts/02_run_debates.py (see those scripts' docstrings) — loading a
second vLLM model into the same process after finishing the first crashes
with a GPU OOM error under tensor parallelism. Run:

    python scripts/05_run_baselines.py --model llama
    python scripts/05_run_baselines.py --model mistral

Usage (after scripts/00_fetch_farm.py and scripts/smoke_test_vllm.py have
completed — this does NOT require scripts/01_build_pools.py, unlike
02_run_debates.py / 03_run_digra_experiments.py, since CoT-SC samples
fresh rather than seeding from pools):
    python scripts/05_run_baselines.py --config configs/base.yaml --model llama
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Standalone-script path fixup (see scripts/04_generate_report.py's
# identical comment): `python scripts/x.py` does not add the project root
# to sys.path the way a notebook implicitly does.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baselines.orchestration import run_baselines_for_dataset_model
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

    registry = RunRegistry(output_root / "checkpoints" / "baseline_registry.json")
    baselines_dir = output_root / "baselines"
    baselines_dir.mkdir(parents=True, exist_ok=True)

    backup_cfg = cfg.backup
    periodic_backup = PeriodicBackup(
        every_n=backup_cfg.every_n_questions,
        local_dest=backup_cfg.local_dest,
        kaggle_dataset_slug=backup_cfg.kaggle_dataset_slug,
        source_dirs=[output_root / "checkpoints", output_root / "baselines"],
    )

    cot_sc_cfg = cfg.cot_sc
    n_samples_max = cot_sc_cfg.n_samples_max
    sc_ks = list(cot_sc_cfg.sc_ks)

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

            out_path = baselines_dir / f"{dataset_key}_{model_name}_cot_sc.jsonl"
            logger.info(
                "Running CoT-SC: dataset=%s model=%s n_samples_max=%d sc_ks=%s -> %s",
                dataset_key, model_name, n_samples_max, sc_ks, out_path,
            )
            counts = run_baselines_for_dataset_model(
                llm=llm,
                dataset_key=dataset_key,
                model_name=model_name,
                df=df,
                seeds=cfg.project.seeds,
                n_samples_max=n_samples_max,
                sc_ks=sc_ks,
                registry=registry,
                out_path=out_path,
                max_tokens=cfg.generation.max_tokens,
                temperature=cfg.generation.temperature,
                top_p=cfg.generation.top_p,
                top_k=cfg.generation.top_k,
                logprobs_topk=cfg.generation.logprobs_topk,
                periodic_backup=periodic_backup,
            )
            logger.info(
                "Done: dataset=%s model=%s — built=%d skipped=%d errors=%d",
                dataset_key, model_name, counts["built"], counts["skipped"], counts["errors"],
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
        help="Optional override YAML merged on top of --config.",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Restrict this run to a single model (e.g. 'llama' or 'mistral'). "
             "Strongly recommended — see module docstring for why running "
             "multiple models in one process can crash with a GPU OOM error.",
    )
    args = parser.parse_args()
    main(args.config, args.overrides, args.model)
