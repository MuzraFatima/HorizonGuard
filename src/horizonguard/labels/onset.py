"""Drift onset from fidelity trajectories, reported as intervals.

Three detectors:

* ``cusum_onset``: one-sided CUSUM for a downward mean shift; the onset is the
  last time the statistic was zero before the alarm.
* ``pelt_changepoints``: PELT with a Gaussian mean-change cost; the onset is
  the first changepoint after which the mean drops.
* ``onset_interval``: a single-changepoint profile likelihood, giving a point
  estimate and the set of changepoints within the chi-square(1) cutoff.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class OnsetInterval:
    estimate: int
    lower: int
    upper: int
    drop: float  # mean before minus mean after

    def contains(self, t: int) -> bool:
        return self.lower <= t <= self.upper


def _noise_scale(x: np.ndarray) -> float:
    """Noise standard deviation from first differences, trimming jumps.

    Differences beyond three robust deviations (the mean shifts themselves)
    are dropped before taking the standard deviation, which is far more
    efficient than the MAD alone on short runs.
    """
    if x.size < 3:
        return 1.0
    d = np.diff(x)
    centre = np.median(d)
    mad = np.median(np.abs(d - centre)) / 0.6745
    if mad <= 1e-12:
        return float(np.std(x) or 1.0)
    kept = d[np.abs(d - centre) <= 3 * mad]
    return float(np.std(kept, ddof=1) / np.sqrt(2)) if kept.size > 1 else float(mad / np.sqrt(2))


def cusum_onset(
    fidelity: np.ndarray, baseline_steps: int = 10, k: float = 0.5, h: float = 8.0
) -> int | None:
    """Onset step from a one-sided lower CUSUM, or None if no alarm.

    ``k`` and ``h`` are in units of the noise scale. The alarm is raised
    causally; the onset itself is then refined offline as the best mean-shift
    split between the last zero of the statistic and the alarm, because a
    pre-onset noise excursion can keep the statistic above zero.
    """
    x = np.asarray(fidelity, dtype=float)
    if x.size <= baseline_steps:
        return None
    baseline = x[:baseline_steps]
    mu0 = float(np.mean(baseline))
    sigma = max(_noise_scale(x), float(np.std(baseline, ddof=1)), 1e-6)
    s = 0.0
    last_zero = 0
    for t, value in enumerate(x):
        s = max(0.0, s + (mu0 - value) / sigma - k)
        if s == 0.0:
            last_zero = t
        elif s > h:
            return _best_split(x, last_zero + 1, t)
    return None


def _best_split(x: np.ndarray, lo: int, hi: int) -> int:
    """Split point in [lo, hi] that minimises the two-segment squared error."""
    best_tau, best_rss = hi, np.inf
    for tau in range(max(lo, 1), min(hi, x.size - 1) + 1):
        rss = np.sum((x[:tau] - x[:tau].mean()) ** 2) + np.sum((x[tau:] - x[tau:].mean()) ** 2)
        if rss < best_rss:
            best_tau, best_rss = tau, rss
    return best_tau


def pelt_changepoints(x: np.ndarray, penalty: float | None = None, min_size: int = 2) -> list[int]:
    """PELT for changes in mean with a Gaussian cost. Returns segment starts after 0."""
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 2 * min_size:
        return []
    sigma2 = _noise_scale(x) ** 2
    if penalty is None:
        penalty = 3.0 * np.log(n)
    cs = np.concatenate([[0.0], np.cumsum(x)])
    cs2 = np.concatenate([[0.0], np.cumsum(x**2)])

    def cost(a: int, b: int) -> float:
        m = b - a
        seg = cs[b] - cs[a]
        return float((cs2[b] - cs2[a] - seg * seg / m) / sigma2)

    best = np.full(n + 1, np.inf)
    best[0] = -penalty
    last = np.zeros(n + 1, dtype=int)
    candidates = [0]
    for t in range(min_size, n + 1):
        options = [(best[s] + cost(s, t) + penalty, s) for s in candidates if t - s >= min_size]
        if not options:
            continue
        value, arg = min(options)
        best[t] = value
        last[t] = arg
        candidates = [s for s in candidates if t - s < min_size or best[s] + cost(s, t) <= best[t]]
        candidates.append(t - min_size + 1)
    points = []
    t = n
    while t > 0:
        s = int(last[t])
        if s > 0:
            points.append(s)
        t = s
    return sorted(points)


def pelt_onset(fidelity: np.ndarray, min_drop: float = 0.05, **kwargs) -> int | None:
    x = np.asarray(fidelity, dtype=float)
    points = pelt_changepoints(x, **kwargs)
    bounds = [0, *points, x.size]
    for i, cp in enumerate(points, start=1):
        before = x[bounds[i - 1] : cp].mean()
        after = x[cp : bounds[i + 1]].mean()
        if before - after >= min_drop:
            return cp
    return None


def onset_interval(
    fidelity: np.ndarray,
    confidence: float = 0.95,
    min_size: int = 2,
    model: str = "shift",
) -> OnsetInterval | None:
    """Profile-likelihood interval for a single downward change.

    ``model="shift"`` fits a step change in mean. ``model="hinge"`` fits a flat
    segment followed by a linear decline, which suits gradual drift. ``drop``
    is the fitted fall in fidelity from onset to the end of the run.
    """
    if model not in {"shift", "hinge"}:
        raise ValueError("model must be 'shift' or 'hinge'")
    x = np.asarray(fidelity, dtype=float)
    n = x.size
    if n < 2 * min_size:
        return None
    sigma2 = _noise_scale(x) ** 2
    taus = np.arange(min_size, n - min_size + 1)
    steps = np.arange(n, dtype=float)
    ll = np.empty(taus.size)
    drops = np.empty(taus.size)
    for i, tau in enumerate(taus):
        if model == "shift":
            m1, m2 = x[:tau].mean(), x[tau:].mean()
            rss = np.sum((x[:tau] - m1) ** 2) + np.sum((x[tau:] - m2) ** 2)
            drops[i] = m1 - m2
        else:
            design = np.column_stack([np.ones(n), np.maximum(0.0, steps - tau)])
            coef, *_ = np.linalg.lstsq(design, x, rcond=None)
            rss = float(np.sum((x - design @ coef) ** 2))
            drops[i] = -coef[1] * (n - 1 - tau)
        ll[i] = -rss / (2 * sigma2)
    valid = drops > 0
    if not valid.any():
        return None
    ll_masked = np.where(valid, ll, -np.inf)
    best = int(np.argmax(ll_masked))
    cutoff = stats.chi2.ppf(confidence, df=1) / 2
    inside = taus[(ll_masked >= ll_masked[best] - cutoff)]
    return OnsetInterval(
        estimate=int(taus[best]),
        lower=int(inside.min()),
        upper=int(inside.max()),
        drop=float(drops[best]),
    )
