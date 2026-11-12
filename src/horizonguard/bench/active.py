"""Variance reduction and active label allocation for fork labels.

* ``shrink`` gives noise-aware benefit labels: a normal-normal empirical-Bayes
  posterior mean per prefix, pulling noisy estimates toward the group mean by
  an amount set by their fork standard error.
* ``control_variate`` is the classical estimator ``mean(y) - beta (mean(c) - mu_c)``
  for a per-seed control with known mean, with beta fitted on the same seeds.
* ``priorities`` and ``top_up`` implement active labelling: spend extra forks
  where the best-arm benefit is most uncertain *and* close to the decision
  threshold, because that is where extra forks change decisions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

from horizonguard.bench.generate import BenchConfig, RunSpec, _fresh, label_prefix
from horizonguard.fork import ForkEngine, run_episode
from horizonguard.spec import Intervention


@dataclass(frozen=True)
class Shrunk:
    posterior_mean: np.ndarray
    posterior_sd: np.ndarray
    prior_mean: float
    prior_var: float


def shrink(estimates: Sequence[float], ses: Sequence[float]) -> Shrunk:
    est = np.asarray(estimates, dtype=float)
    se2 = np.maximum(np.asarray(ses, dtype=float) ** 2, 1e-6)
    mu = float(np.average(est, weights=1 / se2))
    tau2 = max(float(np.var(est, ddof=1) - np.mean(se2)), 0.0) if est.size > 1 else 0.0
    weight = tau2 / (tau2 + se2)
    post_mean = mu + weight * (est - mu)
    post_var = weight * se2
    return Shrunk(post_mean, np.sqrt(post_var), mu, tau2)


def control_variate(y: Sequence[float], c: Sequence[float], mu_c: float) -> tuple[float, float]:
    """Control-variate mean and its standard error."""
    y, c = np.asarray(y, dtype=float), np.asarray(c, dtype=float)
    n = y.size
    if n < 3:
        return float(y.mean()), float(y.std(ddof=1) / np.sqrt(n)) if n > 1 else float("inf")
    var_c = np.var(c, ddof=1)
    beta = float(np.cov(y, c, ddof=1)[0, 1] / var_c) if var_c > 0 else 0.0
    adjusted = y - beta * (c - mu_c)
    return float(adjusted.mean()), float(adjusted.std(ddof=1) / np.sqrt(n))


def best_arm_stats(record: dict[str, Any]) -> tuple[int, float, float]:
    arm = max(record["arms"], key=lambda a: a["benefit"])
    return arm["arm"], arm["benefit"], arm["benefit_se"]


def priorities(records: Sequence[dict[str, Any]], threshold: float = 0.0) -> np.ndarray:
    """Probability that more forks flip the intervene / do-not-intervene decision."""
    out = np.empty(len(records))
    for i, r in enumerate(records):
        _, b, se = best_arm_stats(r)
        se = se if np.isfinite(se) and se > 0 else 0.5 / np.sqrt(r["k"])
        out[i] = 2 * stats.norm.sf(abs(b - threshold) / se)
    return out


def top_up(record: dict[str, Any], cfg: BenchConfig, extra_k: int) -> dict[str, Any]:
    """Re-fork one prefix with ``extra_k`` fresh seeds per arm and merge the labels."""
    spec = RunSpec(
        record["env"], record["agent"], record["drift_type"], 0, record["template"], record["seed"]
    )
    env, agent = _fresh(spec, cfg)
    run_episode(env, agent, spec.seed, run_id=record["run_id"], max_steps=record["t"])
    engine = ForkEngine(record["horizon"])
    snap = engine.snapshot(env, agent, record["run_id"], spec.seed)
    extra = engine.fork(env, agent, snap, tuple(Intervention), k=extra_k, seed_offset=record["k"])
    merged_y = np.concatenate([np.array(record["outcomes"]), extra.outcomes], axis=1)
    extra.outcomes = merged_y
    extra.early_fidelity = np.concatenate(
        [
            np.full((merged_y.shape[0], record["k"], extra.early_fidelity.shape[2]), np.nan),
            extra.early_fidelity,
        ],
        axis=1,
    )
    extra.seeds = [*record["fork_seeds"], *extra.seeds]
    relabelled = label_prefix(extra, record["fidelity_t"])
    # compliance needs both arms' early fidelity; keep the original values
    for new, old in zip(relabelled["arms"], record["arms"], strict=True):
        new["compliance"] = old["compliance"]
    relabelled["fork_steps"] = record["fork_steps"] + extra.steps_run
    return {**record, **relabelled, "k": merged_y.shape[1]}


def allocate(
    records: list[dict[str, Any]],
    cfg: BenchConfig,
    budget_prefixes: int,
    extra_k: int = 6,
    threshold: float = 0.0,
) -> tuple[list[dict[str, Any]], list[int]]:
    """Top up the ``budget_prefixes`` highest-priority prefixes; return records and indices."""
    order = np.argsort(-priorities(records, threshold), kind="stable")[:budget_prefixes]
    updated = list(records)
    for i in order:
        updated[int(i)] = top_up(records[int(i)], cfg, extra_k)
    return updated, [int(i) for i in order]
