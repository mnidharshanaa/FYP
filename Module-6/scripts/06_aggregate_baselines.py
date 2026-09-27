"""
scripts/06_aggregate_baselines.py

Reads results/baselines/*_cot_sc.jsonl (scripts/05_run_baselines.py's
output) and produces:

  1. results/aggregated/baselines_summary.csv — the full Module 1
     checklist table (Accuracy, EM, F1, TruthfulQA metric, Majority Vote
     Share, Vote Entropy, # Unique Answers, Wrong Consensus Rate, Invalid
     Answer Rate, Average Tokens, Total Tokens, Average Latency), one row
     per (dataset, model, N).
  2. results/figures/cotsc_accuracy_vs_n_<dataset>_<model>.png — the
     paper's Fig. 5-style accuracy-vs-sampling-count plot, for every
     (dataset, model) that has data at more than one N.

*** Runs entirely on CPU. No GPU, no vLLM, no Kaggle required. ***
    pip install matplotlib pandas
    python scripts/06_aggregate_baselines.py --baselines-dir results/baselines --out-dir results

If a dataset/model combination has no file, it's silently skipped —
designed to produce a partial, honest report from whatever data you
actually have (e.g. only llama has finished so far).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics.baseline_metrics import aggregate_baseline_records

FILENAME_RE = re.compile(r"^(?P<dataset>[a-z]+)_(?P<model>[a-z]+)_cot_sc\.jsonl$")

FIELDNAMES = [
    "dataset", "model", "n_samples", "n_questions",
    "accuracy", "em", "f1", "truthfulqa_metric",
    "majority_vote_share", "vote_entropy", "n_unique_answers",
    "wrong_consensus_rate", "invalid_answer_rate",
    "average_tokens", "total_tokens", "average_latency_seconds",
]


def discover_files(baselines_dir: str) -> list:
    if not os.path.isdir(baselines_dir):
        return []
    found = []
    for path in sorted(glob.glob(os.path.join(baselines_dir, "*.jsonl"))):
        m = FILENAME_RE.match(os.path.basename(path))
        if not m:
            print(f"  (skipping unrecognized file: {path})")
            continue
        found.append({"dataset": m["dataset"], "model": m["model"], "path": path})
    return found


def load_jsonl(path: str) -> list:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _all_ks(records: list) -> list:
    ks = set()
    for r in records:
        ks.update(int(k) for k in r.get("votes", {}))
    return sorted(ks)


def main(baselines_dir: str, out_dir: str) -> None:
    out_root = Path(out_dir)
    aggregated_dir = out_root / "aggregated"
    figures_dir = out_root / "figures"
    aggregated_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    files = discover_files(baselines_dir)
    if not files:
        print(f"No baseline result files found in {baselines_dir}. Nothing to aggregate.")
        return

    rows = []
    accuracy_by_group: dict = {}  # (dataset, model) -> {k: accuracy}

    for f in files:
        records = load_jsonl(f["path"])
        ks = _all_ks(records)
        if not ks:
            print(f"  (no votes found in {f['path']}, skipping)")
            continue
        for k in ks:
            row = aggregate_baseline_records(records, k)
            if row is None:
                continue
            rows.append(row)
            accuracy_by_group.setdefault((row["dataset"], row["model"]), {})[k] = row["accuracy"]
            print(
                f"  {row['dataset']}/{row['model']} N={k}: "
                f"accuracy={row['accuracy']:.3f} EM={row['em']:.3f} F1={row['f1']:.3f} "
                f"invalid_rate={row['invalid_answer_rate']:.3f}"
            )

    if not rows:
        print("No usable votes across any file. Nothing written.")
        return

    csv_path = aggregated_dir / "baselines_summary.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES + ["_note"])
        writer.writeheader()
        for row in rows:
            writer.writerow({**{k: row.get(k) for k in FIELDNAMES}, "_note": row.get("_note", "")})
    print(f"\nWrote {csv_path} ({len(rows)} rows)")

    _plot_accuracy_vs_n(accuracy_by_group, figures_dir)


def _plot_accuracy_vs_n(accuracy_by_group: dict, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping accuracy-vs-N plot (pip install matplotlib).")
        return

    for (dataset, model), by_k in accuracy_by_group.items():
        if len(by_k) < 2:
            continue  # nothing to show a trend with at a single N
        ks = sorted(by_k)
        accs = [by_k[k] for k in ks]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(ks, accs, marker="o", color="#4C72B0", label="CoT-SC")
        ax.set_xlabel("Number of samples (N)")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)
        ax.set_title(f"CoT-SC accuracy vs. sampling count — {dataset} / {model}")
        ax.legend()
        plt.tight_layout()
        fname = out_dir / f"cotsc_accuracy_vs_n_{dataset}_{model}.png"
        plt.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"  wrote {fname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baselines-dir", type=str, default="results/baselines")
    parser.add_argument("--out-dir", type=str, default="results")
    args = parser.parse_args()
    main(args.baselines_dir, args.out_dir)
