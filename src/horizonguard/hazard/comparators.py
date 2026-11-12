"""Comparators named in the blueprint for Pillar 1.

* ``SingleEventClassifier``: P(violation within H) from step features,
  trained on realised outcomes (no states, no rollout labels).
* ``LogisticHazard``: a single discrete-time violation hazard per step,
  with risk_H = 1 - (1 - h)^H.
* ``PCAAnomaly``: a linear autoencoder fitted on steps of healthy runs;
  the score is reconstruction error ranked against training steps.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from horizonguard.baselines.monitors import Monitor
from horizonguard.embeddings import PrefixFeaturizer
from horizonguard.eval.dataset import PrefixDataset
from horizonguard.hazard.model import step_data_for
from horizonguard.logging import StepRecord, Trajectory
from horizonguard.models import LogisticRegression, Standardizer
from horizonguard.spec import State


def _violation_within(steps_run: np.ndarray, steps_t: np.ndarray, viol_t: np.ndarray, h: int):
    return ((viol_t > 0) & (viol_t - steps_t <= h) & (viol_t - steps_t >= 0)).astype(float)


@dataclass
class SingleEventClassifier(Monitor):
    name: str = "single-event"
    horizon: int = 20
    featurizer: PrefixFeaturizer = field(default_factory=PrefixFeaturizer)
    model: LogisticRegression = field(default_factory=lambda: LogisticRegression(l2=3.0))

    def fit(self, train: PrefixDataset, runs: Mapping[str, Trajectory]) -> SingleEventClassifier:
        d = step_data_for(train, runs)
        # first violated step per run (0 = never), compared with each row's step
        first_v = np.zeros(d.run.max() + 1)
        for i in range(len(d)):
            if d.state[i] == State.VIOLATED and first_v[d.run[i]] == 0:
                first_v[d.run[i]] = d.t[i]
        live = d.state != State.VIOLATED
        y = _violation_within(d.run, d.t, first_v[d.run], self.horizon)
        self.model.fit(d.x[live], y[live])
        return self

    def predict(self, data: PrefixDataset) -> np.ndarray:
        return self.model.predict_proba(data.x)

    def score_steps(self, steps: Sequence[StepRecord]) -> float:
        return float(self.model.predict_proba(self.featurizer(steps)[None, :])[0])


@dataclass
class LogisticHazard(Monitor):
    name: str = "logistic-hazard"
    horizon: int = 20
    featurizer: PrefixFeaturizer = field(default_factory=PrefixFeaturizer)
    model: LogisticRegression = field(default_factory=lambda: LogisticRegression(l2=3.0))

    def fit(self, train: PrefixDataset, runs: Mapping[str, Trajectory]) -> LogisticHazard:
        d = step_data_for(train, runs)
        rows = (d.state != State.VIOLATED) & (d.next_state >= 0)
        self.model.fit(d.x[rows], (d.next_state[rows] == State.VIOLATED).astype(float))
        return self

    def _risk(self, x: np.ndarray) -> np.ndarray:
        h = self.model.predict_proba(x)
        return 1 - (1 - h) ** self.horizon

    def predict(self, data: PrefixDataset) -> np.ndarray:
        return self._risk(data.x)

    def score_steps(self, steps: Sequence[StepRecord]) -> float:
        return float(self._risk(self.featurizer(steps)[None, :])[0])


@dataclass
class PCAAnomaly(Monitor):
    name: str = "autoencoder-anomaly"
    components: int = 8
    featurizer: PrefixFeaturizer = field(default_factory=PrefixFeaturizer)
    scaler: Standardizer = field(default_factory=Standardizer)
    _basis: np.ndarray = field(default_factory=lambda: np.zeros(0))
    _ref: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def _error(self, x: np.ndarray) -> np.ndarray:
        z = self.scaler.transform(np.atleast_2d(x))
        recon = z @ self._basis.T @ self._basis
        return np.sum((z - recon) ** 2, axis=1)

    def fit(self, train: PrefixDataset, runs: Mapping[str, Trajectory]) -> PCAAnomaly:
        d = step_data_for(train, runs)
        bad_runs = {d.run[i] for i in range(len(d)) if d.state[i] == State.VIOLATED}
        healthy = np.array([r not in bad_runs for r in d.run])
        x = d.x[healthy] if healthy.any() else d.x
        z = self.scaler.fit(x).transform(x)
        _, _, vt = np.linalg.svd(z - z.mean(axis=0), full_matrices=False)
        self._basis = vt[: self.components]
        self._ref = np.sort(self._error(x))
        return self

    def _score(self, x: np.ndarray) -> np.ndarray:
        return np.searchsorted(self._ref, self._error(x)) / max(len(self._ref), 1)

    def predict(self, data: PrefixDataset) -> np.ndarray:
        return self._score(data.x)

    def score_steps(self, steps: Sequence[StepRecord]) -> float:
        return float(self._score(self.featurizer(steps)[None, :])[0])
