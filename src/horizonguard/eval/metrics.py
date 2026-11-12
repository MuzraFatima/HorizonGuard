"""Preregistered metrics, computed per run first and then aggregated."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy import stats


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    s, y = np.asarray(scores, dtype=float), np.asarray(labels).astype(bool)
    n_pos, n_neg = y.sum(), (~y).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = stats.rankdata(s)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def brier(prob: Sequence[float], target: Sequence[float]) -> float:
    return float(np.mean((np.asarray(prob) - np.asarray(target)) ** 2))


def ece(prob: Sequence[float], target: Sequence[float], bins: int = 10) -> float:
    """Expected calibration error; ``target`` may be soft (rollout risk)."""
    p, y = np.asarray(prob, dtype=float), np.asarray(target, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def rollout_error(prob: Sequence[float], rollout_risk: Sequence[float]) -> float:
    """Mean absolute gap between predicted risk and fork-measured risk (RQ1)."""
    return float(np.mean(np.abs(np.asarray(prob) - np.asarray(rollout_risk))))


def healthy_mask(length: int, violation_step: int | None, horizon: int) -> np.ndarray:
    """Steps where an alarm counts as false: more than ``horizon`` steps before violation."""
    mask = np.ones(length, dtype=bool)
    if violation_step is not None:
        mask[max(0, violation_step - horizon) :] = False
    return mask


def threshold_at_budget(
    scores_by_run: Sequence[np.ndarray],
    violation_steps: Sequence[int | None],
    budget_per_1000: float,
    horizon: int,
) -> float:
    """Smallest threshold whose false alarms per 1,000 healthy steps stay within budget."""
    healthy = np.concatenate(
        [
            s[healthy_mask(len(s), v, horizon)]
            for s, v in zip(scores_by_run, violation_steps, strict=True)
        ]
    )
    if healthy.size == 0:
        return float("-inf")
    allowed = int(np.floor(budget_per_1000 * healthy.size / 1000))
    ordered = np.sort(healthy)[::-1]
    if allowed >= healthy.size:
        return float(ordered[-1])
    return float(np.nextafter(ordered[allowed], np.inf))


def lead_times(
    scores_by_run: Sequence[np.ndarray],
    violation_steps: Sequence[int | None],
    threshold: float,
) -> np.ndarray:
    """Per violating run: steps between first alarm and violation (0 = missed)."""
    out = []
    for s, v in zip(scores_by_run, violation_steps, strict=True):
        if v is None:
            continue
        alarms = np.flatnonzero(s[:v] >= threshold)
        out.append(v - alarms[0] if alarms.size else 0)
    return np.array(out, dtype=float)


def lead_time_at_budget(
    scores_by_run: Sequence[np.ndarray],
    violation_steps: Sequence[int | None],
    budget_per_1000: float = 5.0,
    horizon: int = 20,
) -> dict[str, float]:
    thr = threshold_at_budget(scores_by_run, violation_steps, budget_per_1000, horizon)
    lt = lead_times(scores_by_run, violation_steps, thr)
    return {
        "threshold": thr,
        "median_lead": float(np.median(lt)) if lt.size else float("nan"),
        "mean_lead": float(lt.mean()) if lt.size else float("nan"),
        "detection_rate": float((lt > 0).mean()) if lt.size else float("nan"),
        "violating_runs": int(lt.size),
    }


def bootstrap_ci(
    groups: Sequence[np.ndarray],
    statistic: Callable[[list[np.ndarray]], float],
    n: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Point estimate and percentile interval, resampling whole runs (groups)."""
    rng = np.random.default_rng(seed)
    point = statistic(list(groups))
    reps = []
    for _ in range(n):
        idx = rng.integers(0, len(groups), len(groups))
        value = statistic([groups[i] for i in idx])
        if np.isfinite(value):
            reps.append(value)
    if not reps:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(reps, [50 * (1 - confidence), 50 * (1 + confidence)])
    return float(point), float(lo), float(hi)
