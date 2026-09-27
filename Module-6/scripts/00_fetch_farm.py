"""
scripts/00_fetch_farm.py

Clones the official Farm dataset repo (Xu et al., ACL 2024) and stages the
4 subset .jsonl files under data/farm/, where src/data/farm_loader.py
expects to find them.

This script contains NO logic beyond fetching/staging — by design, per the
project convention that scripts/ only orchestrates src/ and external tools.

Usage (from repo root):
    python scripts/00_fetch_farm.py
    python scripts/00_fetch_farm.py --dest data/farm --force
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/pillowsofwind/llms-believe-the-earth-is-flat.git"

# Real file names verified directly against the cloned repo
# (src/Farm_dataset/*.jsonl) — not assumed from the README alone.
SUBSET_FILES = {
    "Boolq.jsonl": "Boolq.jsonl",
    "NQ1.jsonl": "NQ1.jsonl",
    "NQ2.jsonl": "NQ2.jsonl",
    "TruthfulQA.jsonl": "TruthfulQA.jsonl",
}

# Sample counts documented in the Farm README — used as a sanity check
# after staging, so a silently-truncated clone is caught immediately
# rather than surfacing later as a confusing pool-building bug.
EXPECTED_LINE_COUNTS = {
    "Boolq.jsonl": 491,
    "NQ1.jsonl": 488,
    "NQ2.jsonl": 489,
    "TruthfulQA.jsonl": 484,
}


def _count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def fetch_farm(dest: Path, force: bool = False, clone_dir: Path | None = None) -> None:
    dest.mkdir(parents=True, exist_ok=True)

    already_staged = all((dest / fname).exists() for fname in SUBSET_FILES.values())
    if already_staged and not force:
        print(f"All FARM subset files already present at {dest}. Use --force to re-fetch.")
        _verify(dest)
        return

    clone_dir = clone_dir or (dest.parent / "_farm_repo_tmp")
    if clone_dir.exists():
        shutil.rmtree(clone_dir)

    print(f"Cloning {REPO_URL} ...")
    subprocess.run(
        ["git", "clone", "--depth", "1", REPO_URL, str(clone_dir)],
        check=True,
    )

    src_dir = clone_dir / "src" / "Farm_dataset"
    if not src_dir.exists():
        raise FileNotFoundError(
            f"Expected dataset directory not found at {src_dir}. "
            f"The upstream repo layout may have changed — check "
            f"{REPO_URL} manually before re-running."
        )

    for src_name, dest_name in SUBSET_FILES.items():
        src_path = src_dir / src_name
        if not src_path.exists():
            raise FileNotFoundError(f"Expected file not found: {src_path}")
        shutil.copy2(src_path, dest / dest_name)
        print(f"  staged {dest_name}")

    shutil.rmtree(clone_dir)
    _verify(dest)


def _verify(dest: Path) -> None:
    print("Verifying line counts against Farm README-documented sample counts...")
    all_ok = True
    for fname, expected in EXPECTED_LINE_COUNTS.items():
        actual = _count_lines(dest / fname)
        status = "OK" if actual == expected else "MISMATCH"
        if actual != expected:
            all_ok = False
        print(f"  {fname}: {actual} lines (expected {expected}) [{status}]")

    if not all_ok:
        print(
            "\nWARNING: line count mismatch detected. This may mean the "
            "upstream dataset changed, or staging was incomplete. "
            "Do not proceed to pool-building until this is resolved.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("All subset files verified.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest", type=Path, default=Path("data/farm"),
        help="Where to stage the 4 .jsonl files (default: data/farm)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-clone and re-stage even if files already exist",
    )
    args = parser.parse_args()
    fetch_farm(dest=args.dest, force=args.force)
