"""
scripts/10_run_propagation_control.py

Runs the Propagation Control Module (the final module) across every
dataset x debate-model x seed combination in the config.

Runs a SETUP SWEEP by default: "standard" (genuine, unseeded generation)
PLUS the controlled-correctness conditions from
configs/base.yaml's debate.hallucination_setups (e.g. "1 agent seeded
wrong, 2 seeded correct") — added per the frozen decision, not replacing
"standard". Controlled setups need scripts/01_build_pools.py to have been
run first for this dataset/model; use --standard-only to skip them (e.g.
for a quick pilot) without needing pools at all.

*** PILOT FIRST. *** configs/base.yaml's propagation_control.n_resample
defaults to 10 (Base 2's own reported setting), per the explicit decision
to measure cost on a small pilot before committing full GPU budget:

    python scripts/10_run_propagation_control.py --model llama \
        --dataset nq --limit-questions 10 --standard-only

Check the logged timing/token/call-count summary this script prints at
the end. If N=10 is too expensive at your actual question count, lower
propagation_control.n_resample in the config (or via --n-resample) to 5
and note the run explicitly as an "adapted implementation," not a silent
substitution.

*** Requires an evidence snapshot for real RAG (recommended). *** Without
one (--no-rag, or simply no snapshot file present for a dataset), every
"verify" decision falls back to the conservative suppress default — this
WILL run and WILL NOT crash, but every verify-triggering edge collapses
to suppress rather than genuinely getting a chance to be rescued:

    python scripts/09_build_evidence_snapshot.py --dataset nq

*** Three experimental conditions, per the frozen design: ***
  Condition A (full method):        default invocation
  Condition B (DIGRA routing only): --bypass-gate
  Condition C (no RAG):             --no-rag

*** IMPORTANT: run ONE model per invocation with --model. *** Same real
Kaggle GPU-OOM failure as every other script in this project under
tensor parallelism.

Usage:
    python scripts/10_run_propagation_control.py --model llama
    python scripts/10_run_propagation_control.py --model mistral
    python scripts/10_run_propagation_control.py --model llama --lam 0.0        # ablation: R-only
    python scripts/10_run_propagation_control.py --model llama --lam 1.0        # ablation: U-only
    python scripts/10_run_propagation_control.py --model llama --bypass-gate    # Condition B
    python scripts/10_run_propagation_control.py --model llama --no-rag         # Condition C
    python scripts/10_run_propagation_control.py --model llama --standard-only  # skip pool-seeded setups
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.orchestration import load_pools_for_dataset_model
from src.data.farm_loader import load_dataset
from src.llm.vllm_client import VLLMClient
from src.rag.retriever import SnapshotRetriever
from src.reliability.orchestration import run_propagation_control_for_dataset_model
from src.utils.backup import PeriodicBackup
from src.utils.checkpoint import RunRegistry
from src.utils.config import load_config
from src.utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


def main(
    config_path: str,
    overrides_path: str,
    model_filter: str,
    dataset_filter: str,
    limit_questions: int,
    lam_override: float,
    no_rag: bool,
    bypass_gate: bool,
    standard_only: bool,
) -> None:
    cfg = load_config(config_path, overrides=overrides_path)
    output_root = Path(cfg.project.output_root)
    setup_logging(output_root)

    registry = RunRegistry(output_root / "checkpoints" / "propagation_control_registry.json")
    out_dir = output_root / "propagation_control"
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir = output_root / "evidence_snapshot"
    pools_dir = output_root / "pools"

    backup_cfg = cfg.backup
    periodic_backup = PeriodicBackup(
        every_n=backup_cfg.every_n_questions,
        local_dest=backup_cfg.local_dest,
        kaggle_dataset_slug=backup_cfg.kaggle_dataset_slug,
        source_dirs=[output_root / "checkpoints", output_root / "propagation_control"],
    )

    pc_cfg = cfg.propagation_control
    lam = lam_override if lam_override is not None else pc_cfg.lam
    setups = ["standard"] if standard_only else ["standard"] + list(cfg.debate.hallucination_setups)
    condition = "B (DIGRA routing only)" if bypass_gate else ("C (no RAG)" if no_rag else "A (full method)")

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
            "Running %d models in a single process — NOT recommended, see module "
            "docstring for the GPU OOM failure this risks. Prefer --model <name>.",
            len(debate_models),
        )

    dataset_keys = [dataset_filter] if dataset_filter else list(cfg.datasets.to_dict())

    for model_name, model_cfg in debate_models.items():
        logger.info("Loading model '%s' (%s)...", model_name, model_cfg["hf_id"])
        llm = VLLMClient(
            model_id=model_cfg["hf_id"], dtype=model_cfg.get("dtype", "bfloat16"),
            max_model_len=model_cfg.get("max_model_len", 4096),
            gpu_memory_utilization=model_cfg.get("gpu_memory_utilization", 0.90),
            tensor_parallel_size=model_cfg.get("tensor_parallel_size", 1),
        )

        for dataset_key in dataset_keys:
            dataset_cfg = cfg.datasets[dataset_key]
            n_questions = limit_questions or dataset_cfg.n_questions
            df = load_dataset(
                dataset_key, farm_dir=cfg.data.farm_dir,
                n_questions=n_questions, seed=cfg.project.seeds[0],
            )

            pools = {}
            if not standard_only:
                pools_path = pools_dir / f"{dataset_key}_{model_name}.jsonl"
                if pools_path.exists():
                    pools = load_pools_for_dataset_model(pools_path)
                else:
                    logger.warning(
                        "No pools found at %s — controlled-correctness setups will be "
                        "skipped for this dataset/model (run scripts/01_build_pools.py "
                        "first, or pass --standard-only to skip them intentionally).",
                        pools_path,
                    )

            retriever = None
            if not no_rag:
                snapshot_path = snapshot_dir / f"{dataset_key}.json"
                if snapshot_path.exists():
                    retriever = SnapshotRetriever(snapshot_path)
                else:
                    logger.warning(
                        "No evidence snapshot found for dataset=%s (%s). Every 'verify' "
                        "decision will fall back to suppress this run. Build it first with "
                        "scripts/09_build_evidence_snapshot.py --dataset %s, or pass --no-rag "
                        "to acknowledge this explicitly and silence this warning.",
                        dataset_key, snapshot_path, dataset_key,
                    )

            out_path = out_dir / (
                f"{dataset_key}_{model_name}_lam{lam}"
                f"{'_bypassgate' if bypass_gate else ''}"
                f"{'_norag' if no_rag else ''}_propagation_control.jsonl"
            )
            logger.info(
                "Running propagation control (Condition %s): dataset=%s model=%s lam=%.2f "
                "n_resample=%d theta=%.2f n_questions=%d setups=%s rag=%s -> %s",
                condition, dataset_key, model_name, lam, pc_cfg.n_resample, pc_cfg.theta,
                len(df), setups, "on" if retriever is not None else "off (fallback=suppress)", out_path,
            )

            start = time.perf_counter()
            counts = run_propagation_control_for_dataset_model(
                llm=llm, dataset_key=dataset_key, model_name=model_name, df=df,
                seeds=cfg.project.seeds, n_agents=cfg.debate.agent_counts[0], n_rounds=cfg.debate.n_rounds,
                lam=lam, tau_p=pc_cfg.tau_p, tau_l=pc_cfg.tau_l,
                n_resample=pc_cfg.n_resample, theta=pc_cfg.theta,
                registry=registry, out_path=out_path,
                setups=setups, pools=pools,
                retriever=retriever, top_k_evidence=pc_cfg.top_k_evidence,
                max_tokens=cfg.generation.max_tokens, temperature=cfg.generation.temperature,
                top_p=cfg.generation.top_p, top_k=cfg.generation.top_k,
                logprobs_topk=cfg.generation.logprobs_topk,
                bypass_reliability_gate=bypass_gate, periodic_backup=periodic_backup,
            )
            duration = time.perf_counter() - start

            logger.info(
                "Done: dataset=%s model=%s — built=%d skipped=%d errors=%d in %.1fs "
                "(%.1fs/question built)",
                dataset_key, model_name, counts["built"], counts["skipped"], counts["errors"],
                duration, duration / max(counts["built"], 1),
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--overrides", type=str, default=None)
    parser.add_argument("--model", type=str, default=None,
                         help="Restrict to one model. Strongly recommended.")
    parser.add_argument("--dataset", type=str, default=None,
                         help="Restrict to one dataset (nq/boolq/truthfulqa). Default: all.")
    parser.add_argument("--limit-questions", type=int, default=None,
                         help="Cap questions per dataset — use this for the N=10 pilot run.")
    parser.add_argument("--lam", type=float, default=None,
                         help="Override configs/base.yaml's propagation_control.lam "
                              "(e.g. for a lambda-sweep ablation).")
    parser.add_argument("--no-rag", action="store_true",
                         help="Condition C: explicitly acknowledge running without an evidence "
                              "snapshot (every verify -> suppress).")
    parser.add_argument("--bypass-gate", action="store_true",
                         help="Condition B: DIGRA routing only — every DIGRA-selected edge "
                              "propagates unconditionally; U/R/PS are still computed and logged "
                              "for comparison, RAG never triggers.")
    parser.add_argument("--standard-only", action="store_true",
                         help="Skip the pool-seeded controlled-correctness setups entirely "
                              "(no scripts/01_build_pools.py needed) — just genuine debate.")
    args = parser.parse_args()
    main(args.config, args.overrides, args.model, args.dataset,
         args.limit_questions, args.lam, args.no_rag, args.bypass_gate, args.standard_only)
