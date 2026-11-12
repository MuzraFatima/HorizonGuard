"""Reproducibility helpers: seeding, config loading and config hashing."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def seed_everything(seed: int) -> np.random.Generator:
    """Seed the global RNGs and return a fresh numpy generator for local use."""
    random.seed(seed)
    np.random.seed(seed % (2**32))  # noqa: NPY002 - legacy global RNG for third-party code
    return np.random.default_rng(seed)


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config at {path} must be a mapping, got {type(data).__name__}")
    return data


def config_hash(config: dict[str, Any], length: int = 12) -> str:
    """Stable short hash of a config, independent of key order."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:length]


def run_record(config: dict[str, Any], seed: int, **extra: Any) -> dict[str, Any]:
    """Metadata every run should log next to its outputs."""
    from horizonguard import __version__

    return {
        "seed": seed,
        "config_hash": config_hash(config),
        "horizonguard_version": __version__,
        **extra,
    }
