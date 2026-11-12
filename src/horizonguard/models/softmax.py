"""Multinomial logistic regression with L2, fitted by L-BFGS."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from horizonguard.models.linear import Standardizer


@dataclass
class SoftmaxRegression:
    """Multinomial logit over a fixed set of classes.

    Classes absent from the training data still get a column, with a large
    negative intercept that is learned like any other parameter. That keeps
    output shapes stable across folds.
    """

    n_classes: int
    l2: float = 1.0
    max_iter: int = 300
    coef_: np.ndarray = field(default_factory=lambda: np.zeros(0))
    scaler: Standardizer = field(default_factory=Standardizer)

    def fit(self, x: np.ndarray, y: np.ndarray, weight: np.ndarray | None = None):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y)
        n, d = x.shape
        k = self.n_classes
        w = np.ones(n) if weight is None else np.asarray(weight, dtype=float)
        xb = np.column_stack([np.ones(n), self.scaler.fit(x).transform(x)])
        onehot = np.zeros((n, k))
        onehot[np.arange(n), y] = 1.0
        mask = np.ones((d + 1, k))
        mask[0] = 0.0  # no penalty on intercepts

        def loss(theta: np.ndarray) -> tuple[float, np.ndarray]:
            b = theta.reshape(d + 1, k)
            logits = xb @ b
            lse = logsumexp(logits, axis=1)
            nll = np.sum(w * (lse - np.sum(onehot * logits, axis=1)))
            prob = np.exp(logits - lse[:, None])
            grad = xb.T @ (w[:, None] * (prob - onehot)) + self.l2 * mask * b
            return nll + 0.5 * self.l2 * np.sum(mask * b**2), grad.ravel()

        res = minimize(
            loss,
            np.zeros((d + 1) * k),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": self.max_iter},
        )
        self.coef_ = res.x.reshape(d + 1, k)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=float))
        logits = np.column_stack([np.ones(len(x)), self.scaler.transform(x)]) @ self.coef_
        return np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
