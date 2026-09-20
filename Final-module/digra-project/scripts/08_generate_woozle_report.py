"""
scripts/08_generate_woozle_report.py

Reproduces Base 1 (Zhang et al., "Beware of the Woozle Effect")'s core
tables/figures from already-saved debate/DIG/DIGRA JSONL results:

  - Table I : MA1-3 / MR2-3 / IMR2-3 / CR2-3, per (dataset, model, method,
              setup) -> results/woozle_report/tables/woozle_table1.csv
  - Fig. 2  : Mean Accuracy at t=1,2,3 vs. first-round Mean Accuracy, one
              chart per (dataset, model, method) that has >=2 setups
              -> results/woozle_report/figures/fig2_<dataset>_<model>_<method>.png
  - Fig. 4  : MA per round, split by question difficulty (median split on
              the pool's `difficulty` field unless --difficulty-threshold
              is given), setup="standard" only, one chart per
              (dataset, model, method) that has standard-setup data AND a
              locatable pool file
              -> results/woozle_report/figures/fig4_<dataset>_<model>_<method>.png

NOT reproduced: Fig. 3 (robustness under repeated misleading
interventions, Appendix B-4) — that requires a separate experiment this
project hasn't implemented; there's no data anywhere this could be
computed from, so it's skipped rather than approximated.

*** Runs entirely on CPU. No GPU, no vLLM required. ***
    pip install matplotlib
    python scripts/08_generate_woozle_report.py --results-dir results

Auto-discovers every relevant file under however many `--results-dir`
roots you give it, however inconsistently nested (handles reorganized/
duplicated export trees) — see src/reports/woozle_report.py's
`discover_result_files` docstring for exactly what does and doesn't
match, and pass --verbose to see every file it skipped and why.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.reports.woozle_report import (
    build_table1_rows,
    compute_fig2_series,
    compute_fig4_data,
    cot_sc_accuracy_reference,
    discover_result_files,
    load_and_group,
    median_split_difficulty,
)

TABLE1_FIELDNAMES = [
    "dataset", "model", "method", "setup", "n_questions",
    "MA1", "MA2", "MA3", "MR2", "MR3", "IMR2", "IMR3", "CR2", "CR3",
]


def _find_first(roots: list, pattern: str) -> str:
    """First file under any root matching pattern (all confirmed
    byte-identical duplicates in practice — see module docstring's
    dedup note — so "first found" is a safe, deterministic choice)."""
    for root in roots:
        matches = sorted(glob.glob(os.path.join(root, "**", pattern), recursive=True))
        if matches:
            return matches[0]
    return None


def main(roots: list, out_dir: str, difficulty_threshold: float, verbose: bool) -> None:
    out_root = Path(out_dir)
    tables_dir = out_root / "tables"
    figures_dir = out_root / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning: {roots}")
    files, skipped = discover_result_files(roots)
    print(f"Discovered {len(files)} usable result files; skipped {len(skipped)}.")
    if verbose:
        for s in skipped:
            print(f"  skip: {s}")
    elif skipped:
        print(f"  (pass --verbose to see all {len(skipped)} skipped files and why)")

    grouped, warnings = load_and_group(files)
    for w in warnings:
        print(f"  WARNING: {w}")

    print("\nData available, by (dataset, model, method):")
    for key, records in sorted(grouped.items()):
        setups = sorted({r.get("setup") for r in records})
        print(f"  {key}: {len(records)} records, setups={setups}")

    # --- Table I ---
    rows = build_table1_rows(grouped)
    csv_path = tables_dir / "woozle_table1.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=TABLE1_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in TABLE1_FIELDNAMES})
    print(f"\nWrote {csv_path} ({len(rows)} rows)")

    # --- Fig. 2 & Fig. 4 need matplotlib; degrade gracefully if absent ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed — skipping Fig.2/Fig.4 (pip install matplotlib).")
        return

    print()
    for (dataset, model, method), records in sorted(grouped.items()):
        _plot_fig2(records, dataset, model, method, roots, figures_dir, plt)
        _plot_fig4(records, dataset, model, method, roots, difficulty_threshold, figures_dir, plt)

    print(f"\nFigures written to {figures_dir}/")
    print(
        "\nNote: Fig. 3 (robustness under repeated misleading interventions, "
        "Appendix B-4) is NOT reproduced — no experiment for it exists yet "
        "in this project's saved results."
    )


def _plot_fig2(records, dataset, model, method, roots, figures_dir, plt) -> None:
    points = compute_fig2_series(records)
    if len(points) < 2:
        return  # nothing to show a trend with at a single setup

    ma1_values = [p[0] for p in points]
    rounds_present = sorted({r for _, _, ma in points for r in ma if ma[r] is not None})

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for r in rounds_present:
        ys = [p[2].get(r) for p in points]
        xs = [x * 100 for x, y in zip(ma1_values, ys) if y is not None]
        ys_pct = [y * 100 for y in ys if y is not None]
        ax.plot(xs, ys_pct, marker="o", label=f"t={r}")

    cot_sc_path = _find_first(roots, f"{dataset}_{model}_cot_sc.jsonl")
    if cot_sc_path:
        ref_acc = cot_sc_accuracy_reference(cot_sc_path, k=5)
        if ref_acc is not None:
            ax.axhline(ref_acc * 100, linestyle="--", color="red",
                       label="model avg accuracy (CoT-SC(5))")

    ax.set_xlabel("Mean Accuracy of First Round (%)")
    ax.set_ylabel("Mean Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"Fig.2-style: {model} / {dataset} / {method}")
    ax.legend()
    plt.tight_layout()
    fname = figures_dir / f"fig2_{dataset}_{model}_{method}.png"
    plt.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  wrote {fname}")


def _plot_fig4(records, dataset, model, method, roots, difficulty_threshold, figures_dir, plt) -> None:
    standard_records = [r for r in records if r.get("setup") == "standard"]
    if not standard_records:
        return

    pool_path = None
    for root in roots:
        matches = sorted(glob.glob(os.path.join(root, "**", "pools", f"{dataset}_{model}.jsonl"), recursive=True))
        if matches:
            pool_path = matches[0]
            break
    if pool_path is None:
        print(f"  (no pool file found for {dataset}/{model} — skipping Fig.4)")
        return

    with open(pool_path) as f:
        pool_records = [json.loads(line) for line in f if line.strip()]
    easy_qids, hard_qids, threshold_used = median_split_difficulty(pool_records, difficulty_threshold)

    data = compute_fig4_data(standard_records, easy_qids, hard_qids)
    if data["n_easy"] == 0 or data["n_hard"] == 0:
        print(f"  (Fig.4 for {dataset}/{model}/{method}: one difficulty bucket is empty, skipping)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, key, title in [
        (axes[0], "easy", f"Easier questions (difficulty >= {threshold_used:.2f}), n={data['n_easy']}"),
        (axes[1], "hard", f"Harder questions (difficulty < {threshold_used:.2f}), n={data['n_hard']}"),
    ]:
        rounds = sorted(r for r in data[key] if data[key][r] is not None)
        ys = [data[key][r] * 100 for r in rounds]
        ax.plot(rounds, ys, marker="o", color="#4C72B0")
        ax.set_xlabel("Debate round")
        ax.set_title(title, fontsize=9)
        ax.set_xticks(rounds)
    axes[0].set_ylabel("Mean Accuracy (%)")
    axes[0].set_ylim(0, 100)
    fig.suptitle(f"Fig.4-style: {model} / {dataset} / {method} (standard setup)")
    plt.tight_layout()
    fname = figures_dir / f"fig4_{dataset}_{model}_{method}.png"
    plt.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  wrote {fname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=str, nargs="+", default=["results"],
                        help="One or more root directories to search recursively (default: results).")
    parser.add_argument("--out-dir", type=str, default="results/woozle_report")
    parser.add_argument("--difficulty-threshold", type=float, default=None,
                        help="Override the median-split threshold for Fig.4 (default: median of the pool).")
    parser.add_argument("--verbose", action="store_true", help="List every skipped file and why.")
    args = parser.parse_args()
    main(args.results_dir, args.out_dir, args.difficulty_threshold, args.verbose)
