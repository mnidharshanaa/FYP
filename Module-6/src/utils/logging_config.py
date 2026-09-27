"""
Centralized logging configuration.

Design intent
-------------
Every script/module calls `get_logger(__name__)` instead of `print()`.
Logs go to both stdout (visible in the Kaggle notebook cell output) and a
persistent log file under the run's output directory, so a Kaggle session
disconnect never loses the record of what happened before the crash.

One log file per run is created, named with a timestamp, so re-running
after a resume doesn't overwrite the previous session's log — both are kept.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

_CONFIGURED = False


def setup_logging(output_dir: str | Path, level: int = logging.INFO) -> Path:
    """
    Configure root logging once per process. Safe to call multiple times;
    only the first call takes effect (subsequent calls are no-ops), so
    scripts can call this defensively without duplicating handlers.

    Returns the path to the log file created.
    """
    global _CONFIGURED

    output_dir = Path(output_dir)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"run_{timestamp}.log"

    if _CONFIGURED:
        logging.getLogger(__name__).warning(
            "setup_logging called again; ignoring (logging already configured). "
            "New log events continue to the original log file for this process."
        )
        return log_path

    root = logging.getLogger()
    root.setLevel(level)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    root.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    _CONFIGURED = True
    logging.getLogger(__name__).info("Logging initialized. Writing to %s", log_path)
    return log_path


def get_logger(name: str) -> logging.Logger:
    """Standard accessor so every module logs under its own module name."""
    return logging.getLogger(name)
