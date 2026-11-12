"""Time-uniform false-alarm control with a betting e-process.

Per-step monitor scores ``x_t`` in [0, 1] are treated as evidence. Under the
null "this run is healthy", assume the conditional mean of the score is at most
``m0``: ``E[x_t | past] <= m0``. Then the wealth

    E_t = prod_{i <= t} (1 + kappa_i (x_i - m0)),   0 <= kappa_i <= 1 / m0

is a non-negative supermartingale whenever each bet ``kappa_i`` is chosen from
scores before step ``i`` (predictable). The bet here is a plug-in: it grows with
the recent excess of scores over ``m0`` and is zero while evidence is weak, so
wealth is not burnt during the healthy part of a run. By Ville's inequality, alarming when
``E_t >= 1 / alpha`` gives ``P(any alarm in a healthy run) <= alpha``, uniformly
over time, however long the run is watched.

``m0`` is estimated from healthy calibration steps as an upper confidence bound
(Hoeffding), so estimation error is accounted for. The guarantee is only as
good as the conditional-mean assumption; ``empirical_run_fpr`` checks it on
held-out healthy runs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass
class BettingEProcess:
    alpha: float = 0.05
    m0: float = 0.2
    kappa: float | None = None  # fixed bet; None means predictable plug-in bets
    window: int = 5
    max_fraction: float = 0.9  # cap on the plug-in bet as a share of 1 / m0

    def __post_init__(self) -> None:
        if not 0 < self.m0 < 1:
            raise ValueError("m0 must lie in (0, 1)")
        if self.kappa is not None and not 0 <= self.kappa <= 1 / self.m0:
            raise ValueError("kappa must lie in [0, 1/m0] to keep wealth non-negative")

    def bets(self, x: np.ndarray) -> np.ndarray:
        if self.kappa is not None:
            return np.full(len(x), self.kappa)
        out = np.zeros(len(x))
        cap = self.max_fraction / self.m0
        for t in range(1, len(x)):
            recent = x[max(0, t - self.window) : t]  # strictly before step t
            excess = recent.mean() - self.m0
            out[t] = np.clip(excess / (self.m0 * (1 - self.m0)), 0.0, cap)
        return out

    @classmethod
    def calibrate(
        cls,
        healthy_scores: Sequence[np.ndarray],
        alpha: float = 0.05,
        confidence: float = 0.95,
        floor: float = 0.02,
    ) -> BettingEProcess:
        """Upper confidence bound on the healthy mean score (Hoeffding over runs)."""
        run_means = np.array([np.mean(s) for s in healthy_scores if len(s)])
        n = max(len(run_means), 1)
        slack = np.sqrt(np.log(1 / (1 - confidence)) / (2 * n))
        m0 = float(np.clip(run_means.mean() + slack, floor, 0.95))
        return cls(alpha=alpha, m0=m0)

    def wealth(self, scores: np.ndarray) -> np.ndarray:
        x = np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)
        return np.cumprod(1.0 + self.bets(x) * (x - self.m0))

    def first_alarm(self, scores: np.ndarray) -> int | None:
        hits = np.flatnonzero(self.wealth(scores) >= 1.0 / self.alpha)
        return int(hits[0]) if hits.size else None


def empirical_run_fpr(proc: BettingEProcess, healthy: Sequence[np.ndarray]) -> float:
    """Share of healthy runs with at least one alarm (target: at most alpha)."""
    return float(np.mean([proc.first_alarm(s) is not None for s in healthy])) if healthy else 0.0


def detection(
    proc: BettingEProcess, scores: Sequence[np.ndarray], violations: Sequence[int | None]
) -> dict[str, float]:
    leads = []
    for s, v in zip(scores, violations, strict=True):
        if v is None:
            continue
        alarm = proc.first_alarm(s[:v])
        leads.append(v - alarm if alarm is not None else 0)
    lt = np.array(leads, dtype=float)
    return {
        "detection_rate": float((lt > 0).mean()) if lt.size else float("nan"),
        "mean_lead": float(lt.mean()) if lt.size else float("nan"),
    }
