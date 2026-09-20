"""
scripts/04_generate_report.py

Reads results/digra/*.jsonl (whatever variants/datasets you actually
have — this does NOT require all 3 variants or all datasets to be
present) and generates comparison figures + a text summary.

*** Runs entirely on CPU. No GPU, no vLLM, no Kaggle required. ***
Run this on your own laptop against your downloaded results/ folder:

    pip install matplotlib pandas
    python scripts/04_generate_report.py --results-dir results/digra --out-dir report

If a variant/dataset combination has no file, it's silently skipped in
that figure, and noted in the printed summary — this is designed to
produce a partial, honest report from whatever data you actually have
(e.g. Llama only, Mistral still pending), not to require everything.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

# Standalone-script path fixup: Jupyter/Kaggle notebooks implicitly add the
# working directory to sys.path, but a plain `python scripts/x.py` does
# NOT — it only adds the script's own directory (scripts/), not the
# project root, so `from src...` imports fail outside a notebook. This
# makes the script work the same way regardless of how it's invoked.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")  # no display needed, just save files
import matplotlib.pyplot as plt

from src.metrics.propagation_metrics import (
    compute_cost_stats,
    compute_final_accuracy,
    compute_propagation_metrics,
    compute_rag_precision_recall,
)
from src.metrics.standard_mad_adapter import adapt_standard_mad_debates

BASELINE_LABEL = "Standard MAD (baseline)"
BASELINE_FILENAME_RE = re.compile(r"^(?P<dataset>[a-z]+)_(?P<model>[a-z]+)_agents(?P<n_agents>\d+)\.jsonl$")


def discover_baseline_files(baseline_dir: str) -> list:
    """Standard MAD files (src/agents/orchestration.py's naming: no variant
    in the filename) — a DIFFERENT pattern than DIGRA's, deliberately kept
    separate so a Standard MAD file is never mistaken for a DIGRA one."""
    if not baseline_dir or not os.path.isdir(baseline_dir):
        return []
    found = []
    for path in sorted(glob.glob(os.path.join(baseline_dir, "*.jsonl"))):
        m = BASELINE_FILENAME_RE.match(os.path.basename(path))
        if not m:
            continue
        found.append({"dataset": m["dataset"], "model": m["model"], "n_agents": int(m["n_agents"]), "path": path})
    return found

VARIANT_ORDER = ["digra", "digra_rag", "digra_rag_memory"]
VARIANT_LABELS = {"digra": "DIGRA", "digra_rag": "DIGRA+RAG", "digra_rag_memory": "DIGRA+RAG+Memory"}
FILENAME_RE = re.compile(r"^(?P<dataset>[a-z]+)_(?P<model>[a-z]+)_(?P<variant>digra(?:_rag)?(?:_memory)?)_agents(?P<n_agents>\d+)\.jsonl$")


def load_jsonl(path: str) -> list:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def discover_result_files(results_dir: str) -> list:
    """Parse every .jsonl in results_dir into (dataset, model, variant, n_agents, path)."""
    found = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.jsonl"))):
        m = FILENAME_RE.match(os.path.basename(path))
        if not m:
            print(f"  (skipping unrecognized file: {path})")
            continue
        found.append({
            "dataset": m["dataset"], "model": m["model"], "variant": m["variant"],
            "n_agents": int(m["n_agents"]), "path": path,
        })
    return found


def load_durations_from_registry(registry_path: str) -> dict:
    """
    The actual per-debate wall-clock duration is recorded in the checkpoint
    registry's metadata (src/digra/orchestration.py's mark_done call), NOT
    in the results .jsonl files themselves. This reads it back out, keyed
    by run_id, so timing plots can be produced from data you already have
    — no rerun needed just to capture timing.

    run_id format (src/utils/checkpoint.py's make_run_id): sorted
    "key=value" pairs joined by "|", e.g.
    "dataset=nq|model=llama|n_agents=3|question_id=nq_0001|seed=0|setup=standard|variant=digra"
    """
    if not os.path.exists(registry_path):
        return {}
    with open(registry_path) as f:
        registry = json.load(f)
    durations = {}
    for run_id, metadata in registry.items():
        if "duration_seconds" in metadata:
            durations[run_id] = metadata["duration_seconds"]
    return durations


def match_duration_for_debate(durations: dict, dataset: str, model: str, variant: str, debate: dict) -> float:
    """Reconstruct the run_id for one debate record and look up its duration."""
    parts = {
        "dataset": dataset, "model": model, "variant": variant,
        "setup": debate["setup"], "seed": None,  # seed isn't stored on DigraDebateResult itself
        "n_agents": debate["n_agents"], "question_id": debate["question_id"],
    }
    # seed isn't part of DigraDebateResult's saved fields, so an exact
    # run_id match isn't always possible from the results file alone —
    # fall back to matching on everything else and taking the first hit,
    # which is correct as long as a given question wasn't run under the
    # same setup with multiple seeds in this dataset (true for the
    # reduced-scope seeds=[0] configuration this project has been using).
    prefix_keys = sorted(k for k in parts if k != "seed")
    for run_id, duration in durations.items():
        fields = dict(kv.split("=", 1) for kv in run_id.split("|"))
        if all(str(parts[k]) == fields.get(k) for k in prefix_keys):
            return duration
    return None


def collect_durations(data: dict, results_dir: str) -> dict:
    """
    For each (dataset, model, variant) group already loaded into `data`,
    attach a list of per-debate durations pulled from that variant's
    checkpoint registry (checkpoints/digra_{variant}_registry.json,
    expected alongside results_dir's parent 'results/' folder).
    """
    results_root = Path(results_dir).parent  # results/digra -> results/
    checkpoints_dir = results_root / "checkpoints"

    duration_map = {}
    for (dataset, model, variant), debates in data.items():
        registry_path = checkpoints_dir / f"digra_{variant}_registry.json"
        durations_by_run_id = load_durations_from_registry(str(registry_path))
        if not durations_by_run_id:
            continue
        matched = []
        for debate in debates:
            d = match_duration_for_debate(durations_by_run_id, dataset, model, variant, debate)
            if d is not None:
                matched.append(d)
        if matched:
            duration_map[(dataset, model, variant)] = matched
    return duration_map


def main(results_dir: str, out_dir: str, baseline_dir: str = None) -> None:
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    files = discover_result_files(results_dir)
    baseline_files = discover_baseline_files(baseline_dir)

    if not files and not baseline_files:
        print(f"No result files found in {results_dir} or {baseline_dir}. Nothing to report.")
        return

    datasets = sorted(set(f["dataset"] for f in files) | set(f["dataset"] for f in baseline_files))
    models = sorted(set(f["model"] for f in files) | set(f["model"] for f in baseline_files))
    print(f"Found data for datasets={datasets}, models={models}")

    data = {}  # (dataset, model, variant) -> list of debate dicts
    for f in files:
        key = (f["dataset"], f["model"], f["variant"])
        data[key] = load_jsonl(f["path"])
        print(f"  loaded {len(data[key])} DIGRA debates: {key}")

    baseline_data = {}  # (dataset, model) -> adapted debate dicts
    for f in baseline_files:
        key = (f["dataset"], f["model"])
        raw = load_jsonl(f["path"])
        baseline_data[key] = adapt_standard_mad_debates(raw)
        print(f"  loaded {len(raw)} Standard MAD (baseline) debates: {key}")

    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    for model in models:
        for dataset in datasets:
            print(f"\n--- {dataset} / {model} ---")
            baseline_key = (dataset, model)
            if baseline_key in baseline_data:
                acc = compute_final_accuracy(baseline_data[baseline_key])
                print(f"  {BASELINE_LABEL:20s}: accuracy={acc:.3f}  n_debates={len(baseline_data[baseline_key])}")
            for variant in VARIANT_ORDER:
                key = (dataset, model, variant)
                if key not in data:
                    print(f"  {VARIANT_LABELS[variant]:20s}: NOT AVAILABLE")
                    continue
                results = data[key]
                acc = compute_final_accuracy(results)
                cost = compute_cost_stats(results)
                rag = compute_rag_precision_recall(results)
                line = f"  {VARIANT_LABELS[variant]:20s}: accuracy={acc:.3f}  n_debates={cost['n_debates']}  "
                line += f"gen_calls/debate={cost['mean_generate_calls']:.1f}  fd_calls/debate={cost['mean_forced_decode_calls']:.1f}"
                if rag["precision"] is not None:
                    line += f"  RAG precision={rag['precision']:.2f} recall={rag['recall']:.2f}"
                    if rag["confident_wrong_recall"] is not None:
                        line += f" confident_wrong_recall={rag['confident_wrong_recall']:.2f}"
                print(line)

    _plot_accuracy_comparison(data, datasets, models, out_dir_path, baseline_data)
    _plot_ma_trajectories(data, datasets, models, out_dir_path, baseline_data)
    _plot_cost_comparison(data, datasets, models, out_dir_path)
    _plot_rag_quality(data, datasets, models, out_dir_path)

    durations = collect_durations(data, results_dir)
    if durations:
        _plot_time_comparison(durations, datasets, models, out_dir_path)
        print(f"\n  (timing data found for {len(durations)} dataset/model/variant group(s) via checkpoint registry)")
    else:
        print(
            "\n  (no checkpoint registry timing data found — timing plot skipped. "
            "This needs results/checkpoints/digra_<variant>_registry.json alongside "
            "your results/digra/ folder; it's created automatically by "
            "scripts/03_run_digra_experiments.py, not something you need to rerun for.)"
        )

    print(f"\nFigures saved to {out_dir_path}/")


def _available_variants(data, dataset, model):
    return [v for v in VARIANT_ORDER if (dataset, model, v) in data]


def _plot_accuracy_comparison(data, datasets, models, out_dir, baseline_data=None):
    baseline_data = baseline_data or {}
    for model in models:
        for dataset in datasets:
            variants = _available_variants(data, dataset, model)
            baseline_key = (dataset, model)
            has_baseline = baseline_key in baseline_data
            if not variants and not has_baseline:
                continue

            labels, accs, colors = [], [], []
            if has_baseline:
                labels.append(BASELINE_LABEL)
                accs.append(compute_final_accuracy(baseline_data[baseline_key]))
                colors.append("#999999")  # visually distinct, clearly "not a DIGRA variant"
            for v in variants:
                labels.append(VARIANT_LABELS[v])
                accs.append(compute_final_accuracy(data[(dataset, model, v)]))
                colors.append("#4C72B0")

            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.bar(labels, accs, color=colors)
            ax.set_ylabel("Final accuracy")
            ax.set_ylim(0, 1)
            ax.set_title(f"Accuracy comparison — {dataset} / {model}")
            for bar, acc in zip(bars, accs):
                ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.02, f"{acc:.2f}", ha="center")
            plt.tight_layout()
            fname = out_dir / f"accuracy_{dataset}_{model}.png"
            plt.savefig(fname, dpi=150)
            plt.close(fig)
            print(f"  wrote {fname}")


def _plot_ma_trajectories(data, datasets, models, out_dir, baseline_data=None):
    baseline_data = baseline_data or {}
    for model in models:
        for dataset in datasets:
            variants = _available_variants(data, dataset, model)
            baseline_key = (dataset, model)
            has_baseline = baseline_key in baseline_data
            if not variants and not has_baseline:
                continue

            fig, ax = plt.subplots(figsize=(6, 4))
            if has_baseline:
                metrics = compute_propagation_metrics(baseline_data[baseline_key])
                rounds = sorted(metrics["MA"].keys())
                values = [metrics["MA"][r] for r in rounds]
                ax.plot(rounds, values, marker="s", linestyle="--", color="#999999", label=BASELINE_LABEL)
            for v in variants:
                metrics = compute_propagation_metrics(data[(dataset, model, v)])
                rounds = sorted(metrics["MA"].keys())
                values = [metrics["MA"][r] for r in rounds]
                ax.plot(rounds, values, marker="o", label=VARIANT_LABELS[v])
            ax.set_xlabel("Debate round")
            ax.set_ylabel("Mean Accuracy (MA)")
            ax.set_title(f"Accuracy over rounds — {dataset} / {model}")
            ax.legend()
            ax.set_ylim(0, 1)
            plt.tight_layout()
            fname = out_dir / f"ma_trajectory_{dataset}_{model}.png"
            plt.savefig(fname, dpi=150)
            plt.close(fig)
            print(f"  wrote {fname}")


def _plot_cost_comparison(data, datasets, models, out_dir):
    for model in models:
        for dataset in datasets:
            variants = _available_variants(data, dataset, model)
            if not variants:
                continue
            gen = [compute_cost_stats(data[(dataset, model, v)])["mean_generate_calls"] for v in variants]
            fd = [compute_cost_stats(data[(dataset, model, v)])["mean_forced_decode_calls"] for v in variants]
            labels = [VARIANT_LABELS[v] for v in variants]

            fig, ax = plt.subplots(figsize=(6, 4))
            x = range(len(labels))
            ax.bar(x, gen, width=0.4, label="generate calls", color="#4C72B0")
            ax.bar([i + 0.4 for i in x], fd, width=0.4, label="forced_decode calls", color="#DD8452")
            ax.set_xticks([i + 0.2 for i in x])
            ax.set_xticklabels(labels)
            ax.set_ylabel("Mean calls per debate")
            ax.set_title(f"Cost comparison — {dataset} / {model}")
            ax.legend()
            plt.tight_layout()
            fname = out_dir / f"cost_{dataset}_{model}.png"
            plt.savefig(fname, dpi=150)
            plt.close(fig)
            print(f"  wrote {fname}")


def _plot_time_comparison(durations: dict, datasets, models, out_dir):
    for model in models:
        for dataset in datasets:
            variants = [v for v in VARIANT_ORDER if (dataset, model, v) in durations]
            if not variants:
                continue
            means = [sum(durations[(dataset, model, v)]) / len(durations[(dataset, model, v)]) for v in variants]
            labels = [VARIANT_LABELS[v] for v in variants]

            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.bar(labels, means, color="#8172B2")
            ax.set_ylabel("Mean wall-clock time per debate (seconds)")
            ax.set_title(f"Timing comparison — {dataset} / {model}")
            for bar, m in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width() / 2, m, f"{m:.1f}s", ha="center", va="bottom")
            plt.tight_layout()
            fname = out_dir / f"timing_{dataset}_{model}.png"
            plt.savefig(fname, dpi=150)
            plt.close(fig)
            print(f"  wrote {fname}")


def _plot_rag_quality(data, datasets, models, out_dir):
    for model in models:
        for dataset in datasets:
            variants = [v for v in ("digra_rag", "digra_rag_memory") if (dataset, model, v) in data]
            if not variants:
                continue
            precisions, recalls, cwr = [], [], []
            for v in variants:
                r = compute_rag_precision_recall(data[(dataset, model, v)])
                precisions.append(r["precision"] or 0)
                recalls.append(r["recall"] or 0)
                cwr.append(r["confident_wrong_recall"] or 0)
            labels = [VARIANT_LABELS[v] for v in variants]

            fig, ax = plt.subplots(figsize=(6, 4))
            x = range(len(labels))
            width = 0.25
            ax.bar([i - width for i in x], precisions, width=width, label="Precision", color="#4C72B0")
            ax.bar(x, recalls, width=width, label="Recall", color="#DD8452")
            ax.bar([i + width for i in x], cwr, width=width, label="Confident-wrong recall", color="#55A868")
            ax.set_xticks(list(x))
            ax.set_xticklabels(labels)
            ax.set_ylim(0, 1)
            ax.set_title(f"RAG contradiction-detector quality — {dataset} / {model}")
            ax.legend()
            plt.tight_layout()
            fname = out_dir / f"rag_quality_{dataset}_{model}.png"
            plt.savefig(fname, dpi=150)
            plt.close(fig)
            print(f"  wrote {fname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=str, default="results/digra")
    parser.add_argument(
        "--baseline-dir", type=str, default="results/debates",
        help="Standard MAD results (src/agents/orchestration.py's output) to plot as a "
             "labeled reference baseline alongside DIGRA variants. Set to '' to disable.",
    )
    parser.add_argument("--out-dir", type=str, default="report")
    args = parser.parse_args()
    main(args.results_dir, args.out_dir, args.baseline_dir or None)
