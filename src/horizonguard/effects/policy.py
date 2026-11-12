"""Effect-aware intervention policy, abstention and the recoverability horizon."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from horizonguard.effects.learners import DecompositionLearner
from horizonguard.eval.dataset import PrefixDataset
from horizonguard.models import LogisticRegression
from horizonguard.prereg import load_preregistration


@dataclass
class EffectAwarePolicy:
    """Choose the arm with the largest predicted net benefit, or do nothing.

    Act at a prefix only when (a) the best predicted net benefit ``r - w d``
    exceeds ``margin`` and (b) the abstention head is not confident that every
    intervention hurts or does nothing.
    """

    weight: float | None = None
    margin: float = 0.0
    abstain_threshold: float = 0.6
    learner: DecompositionLearner = field(default_factory=DecompositionLearner)
    abstain_head: LogisticRegression = field(default_factory=lambda: LogisticRegression(l2=3.0))

    def __post_init__(self) -> None:
        if self.weight is None:
            self.weight = load_preregistration().disruption_weight

    def fit(self, train: PrefixDataset) -> EffectAwarePolicy:
        self.learner.fit(train)
        true_net = train.arm("rescue") - self.weight * train.arm("disruption")
        self.abstain_head.fit(train.x, (true_net.max(axis=1) <= 0).astype(float))
        return self

    def predicted_net(self, data: PrefixDataset) -> np.ndarray:
        return self.learner.predict_net(data, self.weight)

    def abstain_probability(self, data: PrefixDataset) -> np.ndarray:
        return self.abstain_head.predict_proba(data.x)

    def decide(self, data: PrefixDataset) -> np.ndarray:
        net = self.predicted_net(data)
        best = net.argmax(axis=1)
        act = (net.max(axis=1) > self.margin) & (
            self.abstain_probability(data) < self.abstain_threshold
        )
        return np.where(act, best + 1, 0)


def predicted_horizons(
    data: PrefixDataset, benefit: np.ndarray, eps: float
) -> dict[str, int | None]:
    """Per run: last forked step whose best predicted benefit is at least ``eps``."""
    best = benefit.max(axis=1)
    out: dict[str, int | None] = {}
    for run in np.unique(data.meta["run_id"]):
        m = data.meta["run_id"] == run
        steps = np.array([r["t"] for r in data.records])[m]
        ok = steps[best[m] >= eps]
        out[str(run)] = int(ok.max()) if ok.size else None
    return out


def measured_horizons(data: PrefixDataset, eps: float) -> dict[str, int | None]:
    """Per run: fork-measured T_rec(eps) from the paired benefit labels."""
    return predicted_horizons(data, data.arm("benefit"), eps)


def horizon_errors(
    predicted: dict[str, int | None], measured: dict[str, int | None]
) -> dict[str, float]:
    both = [
        (p, m) for r, m in measured.items() if (p := predicted.get(r)) is not None and m is not None
    ]
    agree = np.mean([(predicted.get(r) is None) == (m is None) for r, m in measured.items()])
    errors = np.array([p - m for p, m in both], dtype=float)
    return {
        "runs": len(measured),
        "recoverable_agreement": float(agree),
        "mean_abs_error": float(np.abs(errors).mean()) if errors.size else float("nan"),
        "mean_signed_error": float(errors.mean()) if errors.size else float("nan"),
        "share_early_or_exact": float((errors <= 0).mean()) if errors.size else float("nan"),
    }


def fix_by_countdown(
    steps: Sequence[int], best_benefit: Sequence[float], eps: float, lookback: int = 3
) -> float | None:
    """Steps left before intervening stops helping, by linear extrapolation.

    Returns 0 when the latest predicted benefit is already below ``eps``,
    None when benefit is not falling (no deadline in sight), and otherwise the
    extrapolated number of steps until it crosses ``eps``.
    """
    t = np.asarray(steps, dtype=float)[-lookback:]
    b = np.asarray(best_benefit, dtype=float)[-lookback:]
    if b.size == 0 or b[-1] < eps:
        return 0.0
    if b.size < 2:
        return None
    slope = np.polyfit(t, b, 1)[0]
    if slope >= 0:
        return None
    return float((b[-1] - eps) / -slope)
