"""Small, dependency-free linear models used by baselines and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _add_bias(x: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(x)), x])


@dataclass
class Standardizer:
    mean: np.ndarray = field(default_factory=lambda: np.zeros(0))
    scale: np.ndarray = field(default_factory=lambda: np.ones(0))

    def fit(self, x: np.ndarray) -> Standardizer:
        self.mean = x.mean(axis=0)
        sd = x.std(axis=0)
        self.scale = np.where(sd > 1e-9, sd, 1.0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.scale


@dataclass
class LogisticRegression:
    """L2-regularised logistic regression fitted by Newton's method.

    Targets may be soft labels in [0, 1], so it can be trained directly on
    rollout-grounded risk.
    """

    l2: float = 1.0
    max_iter: int = 50
    tol: float = 1e-8
    coef_: np.ndarray = field(default_factory=lambda: np.zeros(0))
    scaler: Standardizer = field(default_factory=Standardizer)

    def fit(self, x: np.ndarray, y: np.ndarray, weight: np.ndarray | None = None):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        w = np.ones(len(y)) if weight is None else np.asarray(weight, dtype=float)
        xb = _add_bias(self.scaler.fit(x).transform(x))
        beta = np.zeros(xb.shape[1])
        penalty = np.full(xb.shape[1], self.l2)
        penalty[0] = 0.0
        for _ in range(self.max_iter):
            p = 1 / (1 + np.exp(-np.clip(xb @ beta, -30, 30)))
            grad = xb.T @ (w * (p - y)) + penalty * beta
            hess = (xb * (w * p * (1 - p))[:, None]).T @ xb + np.diag(penalty + 1e-9)
            step = np.linalg.solve(hess, grad)
            beta -= step
            if np.max(np.abs(step)) < self.tol:
                break
        self.coef_ = beta
        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        return _add_bias(self.scaler.transform(np.asarray(x, dtype=float))) @ self.coef_

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(self.decision_function(x), -30, 30)))


@dataclass
class RidgeRegression:
    l2: float = 1.0
    coef_: np.ndarray = field(default_factory=lambda: np.zeros(0))
    scaler: Standardizer = field(default_factory=Standardizer)

    def fit(self, x: np.ndarray, y: np.ndarray, weight: np.ndarray | None = None):
        x = np.asarray(x, dtype=float)
        w = np.ones(len(y)) if weight is None else np.asarray(weight, dtype=float)
        xb = _add_bias(self.scaler.fit(x).transform(x))
        penalty = np.full(xb.shape[1], self.l2)
        penalty[0] = 0.0
        a = (xb * w[:, None]).T @ xb + np.diag(penalty)
        self.coef_ = np.linalg.solve(a, (xb * w[:, None]).T @ np.asarray(y, dtype=float))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return _add_bias(self.scaler.transform(np.asarray(x, dtype=float))) @ self.coef_
