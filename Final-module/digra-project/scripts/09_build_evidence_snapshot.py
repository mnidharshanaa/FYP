"""
scripts/09_build_evidence_snapshot.py

ONE-TIME, CPU-only step that needs internet access — run it separately
from the GPU experiment, in its own short session if that's easier on
Kaggle. Fetches candidate evidence documents for every question in a
dataset via Wikipedia's public REST API (src/rag/retriever.py's
WikipediaLiveRetriever) and freezes the results to a local JSON file.

The actual GPU experiment (scripts/10_run_propagation_control.py) reads
ONLY from that frozen file via SnapshotRetriever — no live network calls
happen during the main experiment, so reruns are byte-identical
regardless of Wikipedia's state tomorrow.

Query used per question: the question text itself (not any agent's
later claim — no debate has happened yet when this runs). At verify-time
during the actual experiment, a specific agent's specific claim is
checked against these same pre-fetched, question-relevant documents.

Resumable: re-running skips any question_id already present in the
output file, so a network hiccup partway through doesn't lose earlier
progress and doesn't require flags to avoid re-fetching everything.

Usage:
    python scripts/09_build_evidence_snapshot.py --dataset nq
    python scripts/09_build_evidence_snapshot.py --dataset nq --top-k 3
    python scripts/09_build_evidence_snapshot.py --dataset nq --dataset boolq --dataset truthfulqa
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.farm_loader import load_dataset
from src.rag.retriever import RetrievedDocument, WikipediaLiveRetriever, load_snapshot_raw, save_snapshot
from src.utils.config import load_config
from src.utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


def build_snapshot_for_dataset(
    dataset_key: str,
    df,
    out_path,
    top_k: int,
    request_delay_seconds: float,
    retriever=None,
) -> dict:
    retriever = retriever or WikipediaLiveRetriever()
    existing = load_snapshot_raw(out_path)  # {question_id: [raw dicts]}
    snapshot = {
        qid: [RetrievedDocument(**d) for d in docs] for qid, docs in existing.items()
    }

    n_fetched, n_skipped, n_errors = 0, 0, 0
    for _, row in df.iterrows():
        qid = row["question_id"]
        if qid in snapshot:
            n_skipped += 1
            continue
        try:
            docs = retriever.retrieve(row["question"], top_k=top_k)
            snapshot[qid] = docs
            n_fetched += 1
        except Exception as exc:  # noqa: BLE001 — one bad question must not abort the whole fetch
            logger.error("Evidence fetch failed for question_id=%s: %s", qid, exc)
            n_errors += 1
            continue

        save_snapshot(out_path, snapshot)  # save after every question — cheap, and makes
                                            # this safely interruptible at any point
        if request_delay_seconds > 0:
            time.sleep(request_delay_seconds)

    return {"fetched": n_fetched, "skipped": n_skipped, "errors": n_errors, "total": len(snapshot)}


def main(config_path: str, datasets: list, top_k: int, request_delay_seconds: float) -> None:
    cfg = load_config(config_path)
    output_root = Path(cfg.project.output_root)
    setup_logging(output_root)
    snapshot_dir = output_root / "evidence_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    for dataset_key in datasets:
        dataset_cfg = cfg.datasets[dataset_key]
        df = load_dataset(
            dataset_key, farm_dir=cfg.data.farm_dir,
            n_questions=dataset_cfg.n_questions, seed=cfg.project.seeds[0],
        )
        out_path = snapshot_dir / f"{dataset_key}.json"
        logger.info("Building evidence snapshot: dataset=%s -> %s", dataset_key, out_path)
        counts = build_snapshot_for_dataset(dataset_key, df, out_path, top_k, request_delay_seconds)
        logger.info(
            "Done: dataset=%s — fetched=%d skipped=%d errors=%d (total in snapshot=%d)",
            dataset_key, counts["fetched"], counts["skipped"], counts["errors"], counts["total"],
        )
        if counts["errors"] > 0:
            logger.warning(
                "%d questions in '%s' have NO evidence in the snapshot after this run — "
                "the experiment must handle these as 'no evidence available', not skip "
                "silently. Re-run this script to retry them (already-fetched questions "
                "are skipped, so re-running is cheap).",
                counts["errors"], dataset_key,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument(
        "--dataset", type=str, action="append", dest="datasets",
        help="Dataset to fetch evidence for (nq/boolq/truthfulqa). Repeat for multiple; "
             "default: all three.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Documents to fetch per question.")
    parser.add_argument(
        "--request-delay", type=float, default=0.2,
        help="Seconds to sleep between questions, as a courtesy to Wikipedia's API.",
    )
    args = parser.parse_args()
    datasets = args.datasets or ["nq", "boolq", "truthfulqa"]
    main(args.config, datasets, args.top_k, args.request_delay)
