"""Intervention policies evaluated with fork labels.

A policy looks at a prefix and returns which arm to apply (0 = do nothing).
Net benefit is computed from the counterfactual labels of the chosen arm:
rescue minus ``w`` times disruption, averaged over all prefixes, with zero
for prefixes the policy leaves alone.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from horizonguard.eval.dataset import PrefixDataset
from horizonguard.models import LogisticRegression
from horizonguard.spec import Intervention


def net_benefit(data: PrefixDataset, arms: np.ndarray, weight: float) -> float:
    """Mean over prefixes of rescue - w * disruption for the chosen arm (0 = none)."""
    rescue, disrupt = data.arm("rescue"), data.arm("disruption")
    total = np.zeros(len(data))
    act = arms > 0
    idx = np.flatnonzero(act)
    total[idx] = rescue[idx, arms[idx] - 1] - weight * disrupt[idx, arms[idx] - 1]
    return float(total.mean())


def rescue_disruption(data: PrefixDataset, arms: np.ndarray) -> tuple[float, float, float]:
    """Recovery r = P(rescue | would fail), disruption d = P(disrupt | would succeed), rate."""
    act = arms > 0
    if not act.any():
        return 0.0, 0.0, 0.0
    idx = np.flatnonzero(act)
    rescue = data.arm("rescue")[idx, arms[idx] - 1]
    disrupt = data.arm("disruption")[idx, arms[idx] - 1]
    risk = data.risk[idx]
    r = float(rescue.sum() / max(risk.sum(), 1e-9))
    d = float(disrupt.sum() / max((1 - risk).sum(), 1e-9))
    return r, d, float(act.mean())


@dataclass
class ProbabilityAlarm:
    """Intervene with a fixed arm whenever predicted risk reaches the threshold."""

    threshold: float = 0.5
    arm: Intervention = Intervention.REMINDER

    def decide(self, risk: np.ndarray, data: PrefixDataset) -> np.ndarray:
        return np.where(risk >= self.threshold, int(self.arm), 0)


@dataclass
class PeriodicReminder:
    every: int = 10
    start: int = 2
    arm: Intervention = Intervention.REMINDER

    def decide(self, risk: np.ndarray, data: PrefixDataset) -> np.ndarray:
        t = np.array([r["t"] for r in data.records])
        return np.where((t - self.start) % self.every == 0, int(self.arm), 0)


@dataclass
class ParadoxRule:
    """Intervention Paradox pilot rule: act only if p > d / (r + d) for this agent.

    ``r`` and ``d`` are per-agent recovery and disruption rates estimated from
    training forks, so the rule abstains for agents that interventions hurt.
    """

    arm: Intervention = Intervention.REMINDER
    rates: dict[str, tuple[float, float]] = field(default_factory=dict)

    def fit(self, train: PrefixDataset) -> ParadoxRule:
        everyone = np.full(len(train), int(self.arm))
        agents = train.meta["agent"]
        for a in np.unique(agents):
            r, d, _ = rescue_disruption(train.subset(agents == a), everyone[agents == a])
            self.rates[str(a)] = (r, d)
        return self

    def cutoff(self, agent: str) -> float:
        r, d = self.rates.get(agent, (1.0, 1.0))
        return d / (r + d) if r + d > 0 else 1.0

    def decide(self, risk: np.ndarray, data: PrefixDataset) -> np.ndarray:
        cut = np.array([self.cutoff(str(a)) for a in data.meta["agent"]])
        return np.where(risk > cut, int(self.arm), 0)


@dataclass
class CIRRouter:
    """CIR-style router: separate rescue and harm scores, act when rescue beats harm."""

    weight: float = 2.0
    arm: Intervention = Intervention.REMINDER
    rescue_model: LogisticRegression = field(default_factory=lambda: LogisticRegression(l2=3.0))
    harm_model: LogisticRegression = field(default_factory=lambda: LogisticRegression(l2=3.0))

    def fit(self, train: PrefixDataset) -> CIRRouter:
        i = int(self.arm) - 1
        self.rescue_model.fit(train.x, train.arm("rescue")[:, i])
        self.harm_model.fit(train.x, train.arm("disruption")[:, i])
        return self

    def decide(self, risk: np.ndarray, data: PrefixDataset) -> np.ndarray:
        gain = self.rescue_model.predict_proba(
            data.x
        ) - self.weight * self.harm_model.predict_proba(data.x)
        return np.where(gain > 0, int(self.arm), 0)


def always(arm: Intervention, data: PrefixDataset) -> np.ndarray:
    return np.full(len(data), int(arm))


def never(data: PrefixDataset) -> np.ndarray:
    return np.zeros(len(data), dtype=int)


POLICY_NAMES: Sequence[str] = ("probability-alarm", "periodic", "paradox-rule", "cir-router")
