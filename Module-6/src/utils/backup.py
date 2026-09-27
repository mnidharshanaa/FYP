"""
Backup utilities for long-running Kaggle jobs.

IMPORTANT DISTINCTION, stated explicitly because it caused a real data
loss incident on this project: a copy to another directory under
/kaggle/working is NOT protection against Kaggle session/kernel loss —
it's still the same ephemeral filesystem, and dies with the session just
the same. It only protects against a different failure mode (e.g. a bug
partially corrupting results/ mid-write).

Real protection against session loss requires the data to leave the
session: either a successful download, or a push to a Kaggle Dataset via
the `kaggle` CLI (which persists independently of any notebook kernel).
`kaggle_dataset_backup` below is what actually matters for that;
`local_copy_backup` is a secondary safety net, not a substitute for it.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Union

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def local_copy_backup(source_dirs: list, dest_dir: Union[str, Path]) -> None:
    """
    Fast local copy of `source_dirs` into `dest_dir` (mirrors shutil's
    dirs_exist_ok behavior — overwrites existing files, doesn't fail if
    dest already has a previous backup in it).

    NOT a substitute for kaggle_dataset_backup — see module docstring.
    Never raises: a failed backup should never crash the actual experiment
    it's protecting, so failures are logged and swallowed here.
    """
    dest_dir = Path(dest_dir)
    for source in source_dirs:
        source = Path(source)
        if not source.exists():
            logger.warning("local_copy_backup: source %s does not exist, skipping", source)
            continue
        target = dest_dir / source.name
        try:
            shutil.copytree(source, target, dirs_exist_ok=True)
        except OSError as exc:
            logger.error("local_copy_backup failed for %s -> %s: %s", source, target, exc)


def kaggle_dataset_backup(
    source_dir: Union[str, Path],
    dataset_slug: str,
    version_message: str = "periodic checkpoint backup",
) -> bool:
    """
    Push `source_dir` as a new version of a Kaggle Dataset via the `kaggle`
    CLI — this is what actually survives a session/kernel loss, since
    Kaggle Datasets are stored independently of any running notebook.

    Requires: `pip install kaggle`, a Kaggle API token configured (either
    ~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY env vars — set the
    latter from a Kaggle Secret, the same way HF_TOKEN is set up), and the
    dataset `dataset_slug` (format "username/dataset-name") already
    created once via `kaggle datasets create -p <source_dir>` — this
    function only pushes new *versions* to an existing dataset.

    Returns True on success, False on failure (never raises — same
    reasoning as local_copy_backup: a backup failure must not crash the
    actual experiment run). NOT independently verified in this
    environment (no Kaggle API access here) — verify the first push
    manually on Kaggle before relying on it silently succeeding in a long
    unattended run.
    """
    source_dir = Path(source_dir)
    try:
        result = subprocess.run(
            [
                "kaggle", "datasets", "version",
                "-p", str(source_dir),
                "-m", version_message,
                "--dir-mode", "zip",
            ],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            logger.error(
                "kaggle_dataset_backup failed (exit %d): stdout=%s stderr=%s",
                result.returncode, result.stdout, result.stderr,
            )
            return False
        logger.info("kaggle_dataset_backup succeeded: %s", result.stdout.strip())
        return True
    except (subprocess.SubprocessError, FileNotFoundError, OSError) as exc:
        logger.error("kaggle_dataset_backup raised %s: %s", type(exc).__name__, exc)
        return False


class PeriodicBackup:
    """
    Call `.maybe_backup()` after each unit of progress (e.g. each
    completed question) in a long-running loop. Backs up every
    `every_n` calls — ties backup frequency to real progress, not
    wall-clock time, and requires no second process/cell.
    """

    def __init__(
        self,
        every_n: int,
        local_dest: Union[str, Path, None] = None,
        kaggle_dataset_slug: str = None,
        source_dirs: list = None,
    ):
        self.every_n = every_n
        self.local_dest = local_dest
        self.kaggle_dataset_slug = kaggle_dataset_slug
        self.source_dirs = source_dirs or []
        self._counter = 0

    def maybe_backup(self) -> bool:
        """Returns True if a backup was actually triggered this call."""
        self._counter += 1
        if self.every_n <= 0 or self._counter % self.every_n != 0:
            return False

        logger.info("PeriodicBackup: triggering backup at progress count %d", self._counter)
        if self.local_dest is not None:
            local_copy_backup(self.source_dirs, self.local_dest)
        if self.kaggle_dataset_slug is not None:
            for source in self.source_dirs:
                kaggle_dataset_backup(source, self.kaggle_dataset_slug)
        return True
