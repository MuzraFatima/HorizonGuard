"""Risk monitors: the baselines HorizonGuard must beat.

Every monitor implements ``fit(train, runs)``, ``predict(dataset)`` (risk per
forked prefix) and ``score_steps(steps)`` (a score for any prefix, used for
lead time). All are black-box except ``LatentProbe``, which reads the
simulator's latent drift as the white-box upper bound (the analogue of
activation probes on open-weights agents).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from horizonguard.embeddings import PrefixFeaturizer
from horizonguard.eval.dataset import PrefixDataset
from horizonguard.logging import StepRecord, Trajectory
from horizonguard.models import LogisticRegression


class Monitor:
    name = "monitor"
    white_box = False

    def fit(self, train: PrefixDataset, runs: Mapping[str, Trajectory]) -> Monitor:
        raise NotImplementedError

    def predict(self, data: PrefixDataset) -> np.ndarray:
        raise NotImplementedError

    def score_steps(self, steps: Sequence[StepRecord]) -> float:
        raise NotImplementedError


@dataclass
class FeatureRiskMonitor(Monitor):
    """Logistic risk model on black-box features, trained on rollout risk.

    With ``groups`` it uses only feature groups with those prefixes, which
    gives the trajectory-feature baseline (``act:``, ``short:``) and the
    shortcut-only control (``short:``).
    """

    name: str = "probability"
    groups: tuple[str, ...] | None = None
    l2: float = 3.0
    featurizer: PrefixFeaturizer = field(default_factory=PrefixFeaturizer)
    model: LogisticRegression = field(default_factory=LogisticRegression)
    _cols: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))

    def _select(self, names: list[str]) -> np.ndarray:
        if self.groups is None:
            return np.arange(len(names))
        return np.array([i for i, n in enumerate(names) if n.startswith(self.groups)])

    def fit(self, train: PrefixDataset, runs: Mapping[str, Trajectory]) -> FeatureRiskMonitor:
        self._cols = self._select(train.feature_names)
        self.model = LogisticRegression(l2=self.l2).fit(train.x[:, self._cols], train.risk)
        return self

    def predict(self, data: PrefixDataset) -> np.ndarray:
        return self.model.predict_proba(data.x[:, self._cols])

    def score_steps(self, steps: Sequence[StepRecord]) -> float:
        x = self.featurizer(steps)[self._cols][None, :]
        return float(self.model.predict_proba(x)[0])


@dataclass
class MarkovMonitor(Monitor):
    """SafetyDrift-style absorbing Markov chain over observable action states.

    States are action names plus an absorbing VIOLATED state. Transitions are
    estimated from training runs with add-one smoothing. Risk is the
    probability of absorption within the horizon from the current state.
    """

    name: str = "markov"
    horizon: int = 20
    states: list[str] = field(default_factory=list)
    _absorb: dict[str, float] = field(default_factory=dict)

    def fit(self, train: PrefixDataset, runs: Mapping[str, Trajectory]) -> MarkovMonitor:
        ids = set(train.meta["run_id"])
        seqs = [runs[r] for r in ids if r in runs]
        names = sorted({s.action for run in seqs for s in run.steps})
        self.states = [*names, "VIOLATED"]
        index = {n: i for i, n in enumerate(self.states)}
        n = len(self.states)
        counts = np.ones((n, n))
        for run in seqs:
            for a, b in zip(run.steps, run.steps[1:], strict=False):
                target = "VIOLATED" if b.violated else b.action
                counts[index[a.action], index[target]] += 1
            last = run.steps[-1]
            if last.violated:
                counts[index[last.action], index["VIOLATED"]] += 1
        counts[-1] = 0
        counts[-1, -1] = 1
        p = counts / counts.sum(axis=1, keepdims=True)
        reach = np.linalg.matrix_power(p, self.horizon)[:, -1]
        self._absorb = {s: float(reach[i]) for i, s in enumerate(self.states)}
        return self

    def score_steps(self, steps: Sequence[StepRecord]) -> float:
        if not steps:
            return float(np.mean(list(self._absorb.values())))
        return self._absorb.get(steps[-1].action, float(np.mean(list(self._absorb.values()))))

    def predict(self, data: PrefixDataset) -> np.ndarray:
        raise NotImplementedError("use predict_runs, the chain needs the raw prefix")

    def predict_runs(self, data: PrefixDataset, runs: Mapping[str, Trajectory]) -> np.ndarray:
        return np.array([self.score_steps(runs[r["run_id"]].steps[: r["t"]]) for r in data.records])


@dataclass
class RuleMonitor(Monitor):
    """Rule-based drift monitor: the smoothed distance from the run's own anchor."""

    name: str = "rule"
    feature: str = "geo:dist_ema"
    featurizer: PrefixFeaturizer = field(default_factory=PrefixFeaturizer)
    _col: int = 0

    def fit(self, train: PrefixDataset, runs: Mapping[str, Trajectory]) -> RuleMonitor:
        self._col = train.feature_names.index(self.feature)
        return self

    def predict(self, data: PrefixDataset) -> np.ndarray:
        return np.clip(data.x[:, self._col], 0, 1)

    def score_steps(self, steps: Sequence[StepRecord]) -> float:
        return float(np.clip(self.featurizer(steps)[self._col], 0, 1))


@dataclass
class EValuatorMonitor(Monitor):
    """Sequential likelihood-ratio test on a verifier score (E-valuator style).

    The verifier is the probability monitor's per-step score. Two Beta
    densities are fitted to verifier scores of steps from violating and
    non-violating training runs. The score is log E_t, the running sum of
    log-likelihood ratios. Under the null of a healthy run, E_t is a
    non-negative martingale, so alarming at E_t >= 1/alpha is anytime-valid.
    """

    name: str = "e-valuator"
    verifier: FeatureRiskMonitor = field(default_factory=FeatureRiskMonitor)
    _null: tuple[float, float] = (1.0, 1.0)
    _alt: tuple[float, float] = (1.0, 1.0)

    def fit(self, train: PrefixDataset, runs: Mapping[str, Trajectory]) -> EValuatorMonitor:
        self.verifier.fit(train, runs)
        p = np.clip(self.verifier.predict(train), 1e-4, 1 - 1e-4)
        bad = np.array([r["violation_step"] is not None for r in train.records])
        self._null = self._beta(p[~bad]) if (~bad).sum() > 2 else (1.0, 3.0)
        self._alt = self._beta(p[bad]) if bad.sum() > 2 else (3.0, 1.0)
        return self

    @staticmethod
    def _beta(x: np.ndarray) -> tuple[float, float]:
        m, v = x.mean(), max(x.var(), 1e-4)
        common = max(m * (1 - m) / v - 1, 0.5)
        return float(m * common), float((1 - m) * common)

    def log_lr(self, p: float) -> float:
        p = float(np.clip(p, 1e-4, 1 - 1e-4))
        return float(stats.beta.logpdf(p, *self._alt) - stats.beta.logpdf(p, *self._null))

    def score_steps(self, steps: Sequence[StepRecord]) -> float:
        total = 0.0
        for t in range(1, len(steps) + 1, max(1, len(steps) // 12)):
            total += self.log_lr(self.verifier.score_steps(steps[:t]))
        return total

    def predict(self, data: PrefixDataset) -> np.ndarray:
        # per-prefix probability via the verifier; the sequential score is for alarms
        return self.verifier.predict(data)


@dataclass
class LatentProbe(Monitor):
    """White-box upper bound: logistic risk on the agent's latent drift and step."""

    name: str = "latent-probe"
    white_box: bool = True
    model: LogisticRegression = field(default_factory=LogisticRegression)

    @staticmethod
    def _x(latent: np.ndarray, t: np.ndarray) -> np.ndarray:
        return np.column_stack([latent, latent**2, latent**3, t])

    def fit(self, train: PrefixDataset, runs: Mapping[str, Trajectory]) -> LatentProbe:
        t = np.array([r["t"] for r in train.records], dtype=float)
        self.model = LogisticRegression(l2=0.1).fit(self._x(train.latent, t), train.risk)
        return self

    def predict(self, data: PrefixDataset) -> np.ndarray:
        t = np.array([r["t"] for r in data.records], dtype=float)
        return self.model.predict_proba(self._x(data.latent, t))

    def score_steps(self, steps: Sequence[StepRecord]) -> float:
        latent = np.array([steps[-1].extra.get("latent_drift", 0.0)])
        return float(self.model.predict_proba(self._x(latent, np.array([len(steps)])))[0])


def default_monitors(horizon: int = 20) -> list[Monitor]:
    return [
        FeatureRiskMonitor(name="probability"),
        FeatureRiskMonitor(name="trajectory-features", groups=("act:", "short:")),
        FeatureRiskMonitor(name="shortcut-only", groups=("short:",)),
        MarkovMonitor(horizon=horizon),
        RuleMonitor(),
        EValuatorMonitor(),
        LatentProbe(),
    ]
