"""
Reproduces Base 1 (Zhang et al., "Beware of the Woozle Effect")'s Table I,
Fig. 2, and Fig. 4 from already-saved debate/DIG/DIGRA JSONL results —
pure aggregation, no model calls. See scripts/08_generate_woozle_report.py
for the CLI wrapper and exactly what is/isn't reproduced and why.

Two schemas exist in this project's saved results, and both are handled
here transparently:
  - Standard MAD (src/agents/debate.py's DebateResult): "agent_responses"
    key, adapted via src/metrics/standard_mad_adapter.py before use.
  - DIG/DIGRA (src/digra/digra_debate.py's DigraDebateResult): already
    "agent_records"-shaped, used as-is.
Both share the "setup" field (e.g. "3,0", "standard") this module groups
by, and both feed src/metrics/propagation_metrics.py's MA/MR/IMR/CR
formulas once in the shared {"agent_records": [...]} shape.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Optional

from src.metrics.grading import extract_final_answer, is_correct
from src.metrics.propagation_metrics import compute_propagation_metrics
from src.metrics.standard_mad_adapter import adapt_standard_mad_debate

FILENAME_RE = re.compile(
    r"^(?P<dataset>nq|boolq|truthfulqa)_(?P<model>llama|mistral)_"
    r"(?:(?P<variant>mad_sparse_half|mad_random|dig|digra|digra_rag|digra_rag_memory)_)?"
    r"agents(?P<n_agents>\d+)\.jsonl$"
)

# Table I is specifically about fully-connected Standard MAD vs. DIG/DIGRA
# under the woozle-effect hallucination-seeding sweep — the sparse/random
# MAD baselines are a different (Table-II-style) comparison and are
# deliberately excluded here, not silently mixed in.
OUT_OF_SCOPE_VARIANTS = {"mad_sparse_half", "mad_random"}


@dataclass
class DiscoveredFile:
    path: str
    dataset: str
    model: str
    method: str        # "standard_mad", "dig", "digra", "digra_rag", "digra_rag_memory"
    n_agents: int


def discover_result_files(roots: list) -> tuple:
    """
    Walk every path in `roots` and find every JSONL file that looks like a
    Standard-MAD/DIG/DIGRA debate result, however deeply/inconsistently
    nested (handles the kind of reorganized "module1/mistral-results/",
    "module5/llama/results/" trees a Kaggle export can produce).

    Returns (files: list[DiscoveredFile], skipped: list[str]) — `skipped`
    is a human-readable audit trail of every file this deliberately
    ignored and why (demo/smoke-test artifacts, out-of-scope MAD variants,
    CoT-SC files, or anything that doesn't match the expected schema at
    all), so nothing is silently dropped without a trace.
    """
    files, skipped = [], []
    seen_paths = set()

    for root in roots:
        for dirpath, _dirnames, filenames in os.walk(root):
            if "dig_verification" in dirpath:
                for fn in filenames:
                    if fn.endswith(".jsonl"):
                        skipped.append(f"{os.path.join(dirpath, fn)}: dig_verification smoke-test artifact")
                continue

            for fn in filenames:
                if not fn.endswith(".jsonl"):
                    continue
                path = os.path.join(dirpath, fn)
                real_path = os.path.realpath(path)
                if real_path in seen_paths:
                    skipped.append(f"{path}: duplicate path already discovered, skipping")
                    continue

                m = FILENAME_RE.match(fn)
                if not m:
                    skipped.append(f"{path}: filename doesn't match a known debate-result pattern")
                    continue

                variant = m["variant"]
                if variant in OUT_OF_SCOPE_VARIANTS:
                    skipped.append(f"{path}: '{variant}' is a Table-II baseline, out of scope for Table I")
                    continue
                method = variant if variant else "standard_mad"

                try:
                    with open(path) as f:
                        first_line = f.readline()
                    first_record = json.loads(first_line) if first_line.strip() else {}
                except (OSError, json.JSONDecodeError) as exc:
                    skipped.append(f"{path}: could not read/parse first line ({exc})")
                    continue

                if "samples" in first_record and "votes" in first_record:
                    skipped.append(f"{path}: CoT-SC schema, not a debate result")
                    continue
                if str(first_record.get("question_id", "")).startswith("demo"):
                    skipped.append(f"{path}: demo/smoke-test question_id, not real data")
                    continue
                if "agent_responses" not in first_record and "agent_records" not in first_record:
                    skipped.append(f"{path}: matched filename pattern but body isn't a recognized debate schema")
                    continue

                seen_paths.add(real_path)
                files.append(DiscoveredFile(
                    path=path, dataset=m["dataset"], model=m["model"],
                    method=method, n_agents=int(m["n_agents"]),
                ))

    return files, skipped


def load_and_group(files: list) -> tuple:
    """
    Reads every discovered file and groups raw record dicts by
    (dataset, model, method). Deduplicates on (setup, question_id) within
    a group — every schema here uses a single seed (one line per
    question x setup; see module docstring), so an exact duplicate key is
    always a re-export artifact, never legitimate multi-seed data.
    Records whose (setup, question_id) key repeats with DIFFERING content
    are kept once (first occurrence) but flagged in `warnings`, since
    silently picking one silently discards a real discrepancy worth
    investigating rather than an ordinary duplicate file.

    Returns (grouped: dict[(dataset,model,method) -> list[dict]], warnings: list[str]).
    """
    grouped: dict = {}
    seen_keys: dict = {}
    warnings = []

    for f in files:
        key = (f.dataset, f.model, f.method)
        grouped.setdefault(key, [])
        seen_keys.setdefault(key, {})

        with open(f.path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                dedup_key = (record.get("setup"), record.get("question_id"))

                if dedup_key in seen_keys[key]:
                    if seen_keys[key][dedup_key] != record:
                        warnings.append(
                            f"{f.path}: duplicate (setup={dedup_key[0]}, question_id={dedup_key[1]}) "
                            f"with DIFFERING content — keeping first occurrence, discarding this one"
                        )
                    continue

                seen_keys[key][dedup_key] = record
                grouped[key].append(record)

    return grouped, warnings


def _to_agent_records_shape(record: dict) -> dict:
    """Standard MAD -> agent_records shape; DIG/DIGRA already are one."""
    if "agent_records" in record:
        return record
    return adapt_standard_mad_debate(record)


def build_table1_rows(grouped: dict) -> list:
    """
    Table I: one row per (dataset, model, method, setup), with
    MA1/MA2/MA3, MR2/MR3, IMR2/IMR3, CR2/CR3 (rounds beyond 3 are dropped
    even if present, matching the paper's own 3-round main table; a
    debate that early-stopped before round 3 simply contributes None for
    the rounds it never reached, exactly as
    compute_propagation_metrics already handles).
    """
    rows = []
    for (dataset, model, method), records in sorted(grouped.items()):
        by_setup: dict = {}
        for r in records:
            by_setup.setdefault(r.get("setup"), []).append(r)

        setup_order = ["3,0", "2,1", "1,2", "0,3", "standard"]
        ordered_setups = [s for s in setup_order if s in by_setup] + \
            sorted(s for s in by_setup if s not in setup_order)

        for setup in ordered_setups:
            subset = [_to_agent_records_shape(r) for r in by_setup[setup]]
            metrics = compute_propagation_metrics(subset)
            row = {
                "dataset": dataset, "model": model, "method": method, "setup": setup,
                "n_questions": len(subset),
            }
            for r in (1, 2, 3):
                row[f"MA{r}"] = metrics["MA"].get(r)
                if r >= 2:
                    row[f"MR{r}"] = metrics["MR"].get(r)
                    row[f"IMR{r}"] = metrics["IMR"].get(r)
                    row[f"CR{r}"] = metrics["CR"].get(r)
            rows.append(row)
    return rows


def compute_fig2_series(records: list) -> list:
    """
    Fig. 2 data: for one (dataset, model, method), one point per setup —
    (MA1, {round: MA}) — sorted by the EMPIRICALLY measured MA1 (not the
    nominal x:y ratio), matching the paper's own x-axis definition
    ("Mean Accuracy of First Round(%)", i.e. measured, not assumed).
    """
    by_setup: dict = {}
    for r in records:
        by_setup.setdefault(r.get("setup"), []).append(_to_agent_records_shape(r))

    points = []
    for setup, subset in by_setup.items():
        metrics = compute_propagation_metrics(subset)
        ma1 = metrics["MA"].get(1)
        if ma1 is None:
            continue
        points.append((ma1, setup, metrics["MA"]))
    points.sort(key=lambda p: p[0])
    return points


def median_split_difficulty(pool_records: list, threshold: Optional[float] = None) -> tuple:
    """
    Splits question_ids into "easy" (difficulty >= threshold) and "hard"
    (difficulty < threshold) sets, matching Fig. 4's "(a) Easy questions
    (>= Initial Accuracy%)" / "(b) Complex questions (<= Initial
    Accuracy%)" split. The paper doesn't give an exact numeric threshold
    beyond "a predefined threshold" — this defaults to the MEDIAN
    difficulty across the given pool, which is a reasonable, honestly-
    documented choice rather than a guess at the paper's unstated exact
    cutoff. Pass `threshold` explicitly to override.

    Returns (easy_qids: set, hard_qids: set, threshold_used: float).
    """
    if not pool_records:
        raise ValueError("pool_records is empty — cannot compute a difficulty split")

    difficulties = sorted(r["difficulty"] for r in pool_records)
    if threshold is None:
        n = len(difficulties)
        threshold = (
            difficulties[n // 2] if n % 2 == 1
            else (difficulties[n // 2 - 1] + difficulties[n // 2]) / 2
        )

    easy_qids = {r["question_id"] for r in pool_records if r["difficulty"] >= threshold}
    hard_qids = {r["question_id"] for r in pool_records if r["difficulty"] < threshold}
    return easy_qids, hard_qids, threshold


def compute_fig4_data(standard_setup_records: list, easy_qids: set, hard_qids: set) -> dict:
    """
    Fig. 4 data: MA per round (1..3), computed separately for the "easy"
    and "hard" question_id sets, restricted to setup="standard" records
    (matching the paper's own scope for this figure — it studies genuine
    debate behavior stratified by difficulty, not the artificially
    hallucination-seeded setups).

    Returns {"easy": {round: MA}, "hard": {round: MA}, "n_easy": int, "n_hard": int}.
    """
    easy_subset = [
        _to_agent_records_shape(r) for r in standard_setup_records
        if r.get("question_id") in easy_qids
    ]
    hard_subset = [
        _to_agent_records_shape(r) for r in standard_setup_records
        if r.get("question_id") in hard_qids
    ]
    result = {"n_easy": len(easy_subset), "n_hard": len(hard_subset)}
    result["easy"] = compute_propagation_metrics(easy_subset)["MA"] if easy_subset else {}
    result["hard"] = compute_propagation_metrics(hard_subset)["MA"] if hard_subset else {}
    return result


def cot_sc_accuracy_reference(cot_sc_path: str, k: int = 5) -> Optional[float]:
    """
    Loads a CoT-SC results file (scripts/05_run_baselines.py's output) and
    returns its accuracy at N=k — used as Fig. 2's dashed "model average
    accuracy on this dataset" reference line when available. Returns None
    (never a fabricated 0.0) if the file doesn't exist or has no vote at
    that k, so the caller can skip the reference line honestly.
    """
    from src.metrics.baseline_metrics import aggregate_baseline_records

    if not os.path.isfile(cot_sc_path):
        return None
    records = []
    with open(cot_sc_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    agg = aggregate_baseline_records(records, k)
    return agg["accuracy"] if agg else None
