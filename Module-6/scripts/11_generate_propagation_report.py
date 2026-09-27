"""
scripts/11_generate_propagation_report.py

Reads scripts/10_run_propagation_control.py's output
(results/propagation_control/*.jsonl — one or more files, e.g. one per
--lam value you ran) and produces:

  - results/propagation_control_report/lambda_ablation.csv   (Section 24)
  - results/propagation_control_report/four_quadrant.csv     (Section 10)
  - results/propagation_control_report/rag_effect_summary.txt (Section 13)

*** Runs entirely on CPU. No GPU, no vLLM required. ***
    python scripts/11_generate_propagation_report.py

NOT YET built (disclosed, not hidden):
  - PS-validity binning (Section 23) — bins communications by PS decile
    and checks whether higher PS actually correlates with better
    downstream correctness, a sanity check on the score itself.
  - The calibration/test-split machinery (Section 7) — right now every
    row in these tables is computed over whatever you passed to
    scripts/10, with no calibration/test separation enforced. If you're
    choosing lambda/thresholds FROM this report's output, freeze your
    choice on a calibration subset before evaluating on a held-out test
    subset — this script does not do that split for you yet.
"""

from __future__ import annotations

import csv
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.reports.propagation_report import (
    four_quadrant_table,
    lambda_ablation_table,
    load_propagation_records,
    rag_effect_summary,
)

LAMBDA_FIELDNAMES = [
    "lambda", "n_debates", "accuracy", "MA1", "MA_final", "MR_final", "IMR_final",
    "WWR", "WCR", "CWR", "n_candidate_edges", "rag_trigger_rate", "avg_tokens_partial",
]
QUADRANT_FIELDNAMES = [
    "quadrant", "n_edges", "pct_of_total", "accuracy_after_propagation",
    "harmful_propagation_rate", "WWR", "CWR", "rag_invocation_count",
]


def main(results_dir: str, out_dir: str) -> None:
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    paths = sorted(glob.glob(str(Path(results_dir) / "*.jsonl")))
    if not paths:
        print(f"No propagation-control result files found in {results_dir}. Nothing to report.")
        print("Run scripts/10_run_propagation_control.py first.")
        return

    print(f"Loading {len(paths)} file(s):")
    for p in paths:
        print(f"  {p}")
    debates = load_propagation_records(paths)
    print(f"Loaded {len(debates)} debate records.\n")

    lambda_rows = lambda_ablation_table(debates)
    lambda_path = out_root / "lambda_ablation.csv"
    with lambda_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LAMBDA_FIELDNAMES)
        writer.writeheader()
        for row in lambda_rows:
            writer.writerow({k: row.get(k) for k in LAMBDA_FIELDNAMES})
    print(f"Wrote {lambda_path} ({len(lambda_rows)} rows)")
    for row in lambda_rows:
        print(
            f"  lambda={row['lambda']}: accuracy={row['accuracy']:.3f} "
            f"WWR={row['WWR']} WCR={row['WCR']} rag_rate={row['rag_trigger_rate']}"
        )

    quadrant_rows = four_quadrant_table(debates)
    quadrant_path = out_root / "four_quadrant.csv"
    with quadrant_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=QUADRANT_FIELDNAMES)
        writer.writeheader()
        for row in quadrant_rows:
            writer.writerow({k: row.get(k) for k in QUADRANT_FIELDNAMES})
    print(f"\nWrote {quadrant_path} ({len(quadrant_rows)} rows)")
    for row in quadrant_rows:
        print(
            f"  {row['quadrant']:12} n={row['n_edges']:4} "
            f"({row['pct_of_total']:.1%}) accuracy_after={row['accuracy_after_propagation']} "
            f"harmful_rate={row['harmful_propagation_rate']}"
        )

    rag_summary = rag_effect_summary(debates)
    rag_path = out_root / "rag_effect_summary.txt"
    with rag_path.open("w") as f:
        if rag_summary is None:
            f.write("No RAG events found in this data — 'verify' was never triggered, "
                    "or no data included a retriever.\n")
        else:
            for k, v in rag_summary.items():
                f.write(f"{k}: {v}\n")
    print(f"\nWrote {rag_path}")
    if rag_summary is None:
        print("  (no RAG events found in this data)")
    else:
        for k, v in rag_summary.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=str, default="results/propagation_control")
    parser.add_argument("--out-dir", type=str, default="results/propagation_control_report")
    args = parser.parse_args()
    main(args.results_dir, args.out_dir)
