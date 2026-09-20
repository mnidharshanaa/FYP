"""
Deterministic seeding.

Why this exists as its own module
----------------------------------
The project reuses the *same* 4 seeds (configs/base.yaml -> project.seeds)
across every dataset/model/method combination, specifically so that
DIGRA and Standard MAD start from identical round-1 states (mirrors
Appendix B-1 of the DIGRA paper, which does this to remove sampling
randomness as a confound). Getting seeding wrong silently invalidates
that comparison, so it is centralized here and logged every time it's set.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def set_global_seed(seed: int) -> None:
    """Seed every RNG source we know about. Call once at the start of every
    (dataset, model, method, seed) run — not once globally at program start —
    so runs are independently reproducible even if re-executed individually."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if _HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    logger.info("Global seed set to %d", seed)
