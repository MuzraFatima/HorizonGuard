"""Featurised view of DriftBench-CF for modelling.

``PrefixDataset`` has one row per forked prefix (fork labels available).
``StepScores`` helpers featurise every step of every base run, which lead-time
metrics need. Features are cached next to the benchmark, keyed by the
featurizer settings, because featurisation dominates evaluation time.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from horizonguard.bench.generate import load_prefixes
from horizonguard.embeddings import PrefixFeaturizer
from horizonguard.logging import Trajectory, group_runs, read_jsonl

ARM_NAMES = ("reminder", "restatement", "gate", "escalation")


@dataclass
class PrefixDataset:
    x: np.ndarray
    feature_names: list[str]
    records: list[dict[str, Any]]
    latent: np.ndarray  # white-box drift at the prefix (probe upper bound only)
    meta: dict[str, np.ndarray] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.records)

    @property
    def risk(self) -> np.ndarray:
        return np.array([r["risk"] for r in self.records])

    @property
    def k(self) -> np.ndarray:
        return np.array([r["k"] for r in self.records])

    def arm(self, field_name: str) -> np.ndarray:
        """(n, 4) matrix of a per-arm label, e.g. ``benefit`` or ``disruption``."""
        return np.array([[a[field_name] for a in r["arms"]] for r in self.records])

    def column(self, key: str) -> np.ndarray:
        return np.array([r[key] for r in self.records], dtype=object)

    def subset(self, mask: np.ndarray) -> PrefixDataset:
        idx = np.flatnonzero(mask)
        return PrefixDataset(
            self.x[idx],
            self.feature_names,
            [self.records[i] for i in idx],
            self.latent[idx],
            {k: v[idx] for k, v in self.meta.items()},
        )

    def select_runs(self, run_ids: Sequence[str]) -> PrefixDataset:
        ids = set(run_ids)
        return self.subset(np.array([r["run_id"] in ids for r in self.records]))

    def drop_groups(self, prefixes: Sequence[str]) -> np.ndarray:
        keep = [i for i, n in enumerate(self.feature_names) if not n.startswith(tuple(prefixes))]
        return self.x[:, keep]


def load_runs(bench_dir: str | Path) -> dict[str, Trajectory]:
    return {t.run_id: t for t in group_runs(read_jsonl(Path(bench_dir) / "trajectories.jsonl.gz"))}


def _cache_key(featurizer: PrefixFeaturizer, bench_dir: Path) -> str:
    manifest = (bench_dir / "manifest.json").read_text(encoding="utf-8")
    settings = json.dumps(
        {
            "dim": featurizer.encoder.dim,
            "anchor": featurizer.anchor_steps,
            "window": featurizer.window,
            "ema": featurizer.ema,
            "buckets": featurizer.action_buckets,
            "names": featurizer.feature_names(),
        },
        sort_keys=True,
    )
    return hashlib.sha256((manifest + settings).encode()).hexdigest()[:16]


def build_prefix_dataset(
    bench_dir: str | Path,
    featurizer: PrefixFeaturizer | None = None,
    use_cache: bool = True,
) -> PrefixDataset:
    bench_dir = Path(bench_dir)
    featurizer = featurizer or PrefixFeaturizer()
    records = load_prefixes(bench_dir)
    cache = bench_dir / f"features-{_cache_key(featurizer, bench_dir)}.npz"
    if use_cache and cache.exists():
        data = np.load(cache)
        x, latent = data["x"], data["latent"]
    else:
        runs = load_runs(bench_dir)
        x = np.array([featurizer(runs[r["run_id"]].steps[: r["t"]]) for r in records])
        latent = np.array(
            [runs[r["run_id"]].steps[r["t"] - 1].extra.get("latent_drift", 0.0) for r in records]
        )
        if use_cache:
            np.savez_compressed(cache, x=x, latent=latent)
    meta = {k: np.array([r[k] for r in records]) for k in ("agent", "family", "env", "run_id")}
    return PrefixDataset(x, featurizer.feature_names(), records, latent, meta)


def step_scores(
    runs: Sequence[Trajectory],
    score: Callable[[Sequence], float],
    stride: int = 1,
) -> list[np.ndarray]:
    """Score every ``stride``-th prefix of each run (others carry the last score forward)."""
    out = []
    for run in runs:
        s = np.empty(len(run))
        last = 0.0
        for t in range(len(run)):
            if t % stride == 0:
                last = score(run.steps[: t + 1])
            s[t] = last
        out.append(s)
    return out
