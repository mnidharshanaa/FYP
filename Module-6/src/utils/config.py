"""
Config loading utility.

Design intent
-------------
Every other module in this project receives a `Config` object, never a raw
path string or a hardcoded constant. This is what makes an ablation a
one-line YAML edit instead of a code change, which matters both for
correctness (no silently-stale hardcoded values) and for reproducibility
(every run can be reconstructed exactly from the config file it used).

Usage
-----
    from src.utils.config import load_config
    cfg = load_config("configs/base.yaml", overrides="configs/ablation_no_rag.yaml")
    cfg.digra.alpha            # dot access
    cfg["digra"]["alpha"]      # dict-style access also works
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Optional, Union

import yaml


class Config:
    """Thin wrapper giving dict-of-dicts YAML content dot-access, recursively."""

    def __init__(self, data: dict):
        object.__setattr__(self, "_data", data)

    def __getattr__(self, key: str) -> Any:
        try:
            value = self._data[key]
        except KeyError as exc:
            raise AttributeError(
                f"Config has no field '{key}'. Available: {list(self._data.keys())}"
            ) from exc
        return self._wrap(value)

    def __getitem__(self, key: str) -> Any:
        return self._wrap(self._data[key])

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __repr__(self) -> str:
        return f"Config({self._data!r})"

    def to_dict(self) -> dict:
        """Return the plain (unwrapped) dict, for logging/serialization."""
        return copy.deepcopy(self._data)

    @staticmethod
    def _wrap(value: Any) -> Any:
        if isinstance(value, dict):
            return Config(value)
        if isinstance(value, list):
            return [Config(v) if isinstance(v, dict) else v for v in value]
        return value


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into `base`, returning a new dict."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(
    base_path: Union[str, Path],
    overrides: Optional[Union[str, Path]] = None,
) -> Config:
    """
    Load `base_path` YAML, optionally deep-merging an `overrides` YAML on top
    (used for ablations, e.g. configs/ablation_no_rag.yaml disabling rag.enabled).
    """
    base_path = Path(base_path)
    with base_path.open("r") as f:
        data = yaml.safe_load(f)

    if overrides is not None:
        overrides_path = Path(overrides)
        with overrides_path.open("r") as f:
            override_data = yaml.safe_load(f)
        data = _deep_merge(data, override_data)

    return Config(data)
