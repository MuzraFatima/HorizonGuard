"""Estimators for the quantities in the estimand sheet (docs/spec.md, section 8).

Fork outcomes for one prefix are stored as a binary matrix ``Y`` of shape
``(n_arms, K)``. Row 0 is the no-intervention arm. Column ``k`` holds
continuations that share seed ``k`` across arms (common random numbers), so
each column is one unit with a full set of potential outcomes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

Array = np.ndarray


def _as_outcomes(y: Array | Sequence[Sequence[int]]) -> Array:
    arr = np.asarray(y)
    if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] < 1:
        raise ValueError("fork outcomes must have shape (n_arms, K)")
    if not np.isin(arr, (0, 1)).all():
        raise ValueError("fork outcomes must be binary")
    return arr.astype(np.int8)


@dataclass(frozen=True)
class Estimate:
    value: float
    se: float
    lower: float
    upper: float

    @property
    def excludes_zero(self) -> bool:
        return self.lower > 0 or self.upper < 0


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    if n <= 0:
        raise ValueError("n must be positive")
    z = stats.norm.ppf(0.5 + confidence / 2)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def arm_risk(y: Array, arm: int = 0, confidence: float = 0.95) -> Estimate:
    """P(Y_t(s) = 1) for one arm, with a Wilson interval."""
    y = _as_outcomes(y)
    row = y[arm]
    k = row.size
    p = float(row.mean())
    lo, hi = wilson_interval(int(row.sum()), k, confidence)
    return Estimate(p, float(np.sqrt(p * (1 - p) / k)), lo, hi)


def _normal(value: float, se: float, confidence: float) -> Estimate:
    z = stats.norm.ppf(0.5 + confidence / 2)
    return Estimate(value, se, value - z * se, value + z * se)


def benefit(y: Array, arm: int, confidence: float = 0.95) -> Estimate:
    """B_t(s) = P(Y(none)=1) - P(Y(s)=1), estimated by the paired mean difference."""
    y = _as_outcomes(y)
    diff = y[0].astype(float) - y[arm].astype(float)
    k = diff.size
    se = float(diff.std(ddof=1) / np.sqrt(k)) if k > 1 else float("inf")
    return _normal(float(diff.mean()), se, confidence)


@dataclass(frozen=True)
class EffectDecomposition:
    rescue: float
    disruption: float
    both_fail: float
    both_succeed: float
    k: int

    @property
    def benefit(self) -> float:
        return self.rescue - self.disruption

    def net_benefit(self, weight: float) -> float:
        return self.rescue - weight * self.disruption

    def net_benefit_estimate(self, weight: float, confidence: float = 0.95) -> Estimate:
        """Delta-method interval for rescue - w * disruption (multinomial counts)."""
        r, d = self.rescue, self.disruption
        var = (r * (1 - r) + weight**2 * d * (1 - d) + 2 * weight * r * d) / self.k
        return _normal(self.net_benefit(weight), float(np.sqrt(max(var, 0.0))), confidence)


def decompose(y: Array, arm: int) -> EffectDecomposition:
    """Rescue and disruption rates from seed-paired fork outcomes."""
    y = _as_outcomes(y)
    none, treated = y[0], y[arm]
    k = none.size
    return EffectDecomposition(
        rescue=float(np.mean((none == 1) & (treated == 0))),
        disruption=float(np.mean((none == 0) & (treated == 1))),
        both_fail=float(np.mean((none == 1) & (treated == 1))),
        both_succeed=float(np.mean((none == 0) & (treated == 0))),
        k=k,
    )


def best_arm(y: Array, weight: float = 0.0) -> tuple[int, float]:
    """Arm with the highest net benefit (weight 0 means plain benefit). Arm 0 scores 0."""
    y = _as_outcomes(y)
    scores = [0.0] + [decompose(y, s).net_benefit(weight) for s in range(1, y.shape[0])]
    idx = int(np.argmax(scores))
    return idx, float(scores[idx])


def recoverability_horizon(
    forks_by_step: Mapping[int, Array],
    eps: float,
    use_lower_bound: bool = True,
    confidence: float = 0.95,
) -> int | None:
    """T_rec(eps): last step t with max_s B_t(s) >= eps.

    By default the lower confidence bound of each B_t(s) is used, so the
    estimate errs early. Returns None when no step qualifies.
    """
    qualifying = []
    for t, y in forks_by_step.items():
        y = _as_outcomes(y)
        if y.shape[0] < 2:
            continue
        best = max(
            (benefit(y, s, confidence).lower if use_lower_bound else benefit(y, s).value)
            for s in range(1, y.shape[0])
        )
        if best >= eps:
            qualifying.append(t)
    return max(qualifying) if qualifying else None


def fix_by_countdown(current_step: int, horizon: int | None) -> int | None:
    """Steps left before intervening stops helping; negative means too late."""
    return None if horizon is None else horizon - current_step
