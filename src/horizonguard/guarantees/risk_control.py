"""Certifying the disruption risk of an alert policy (Pillar 3).

A policy ``pi_lambda`` acts at a prefix when its score (predicted net benefit)
is at least ``lambda``, choosing the arm with the largest predicted net
benefit. Its loss at a prefix is the fork-estimated disruption probability of
the chosen arm, or 0 when it does not act. The loss is bounded in [0, 1] and
unbiased for the true disruption probability.

Prefixes within a run are dependent, so the exchangeable unit is the *run*:
``L_j(lambda)`` is the mean loss over run ``j``'s prefixes. ``R(lambda)`` is the
expected run-level loss, which is non-increasing in ``lambda``.

* ``crc_threshold``: conformal risk control. Pick the smallest ``lambda`` with
  ``n/(n+1) R_n(lambda) + 1/(n+1) <= alpha``, which gives ``E[R] <= alpha``.
* ``ltt_threshold``: Learn-then-Test. Starting from the most conservative
  ``lambda``, test ``H0: R(lambda) > alpha`` with Hoeffding-Bentkus p-values in a
  fixed sequence and stop at the first non-rejection. With probability at
  least ``1 - delta`` the selected threshold has ``R <= alpha``. Fixed-sequence
  testing stays valid even when ``R`` is not monotone.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

NEVER = float("inf")


def run_losses(
    scores: np.ndarray,
    losses: np.ndarray,
    run_ids: Sequence[str],
    lambdas: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """(n_runs, n_lambdas) mean loss per run when acting at score >= lambda."""
    runs = sorted(set(run_ids))
    index = {r: i for i, r in enumerate(runs)}
    ids = np.array([index[r] for r in run_ids])
    out = np.zeros((len(runs), len(lambdas)))
    counts = np.bincount(ids, minlength=len(runs))
    for j, lam in enumerate(lambdas):
        acted = np.where(scores >= lam, losses, 0.0)
        out[:, j] = np.bincount(ids, weights=acted, minlength=len(runs)) / counts
    return out, runs


def hoeffding_bentkus_p(mean_loss: float, n: int, alpha: float) -> float:
    """p-value for H0: R > alpha given an empirical mean of n losses in [0, 1]."""
    if mean_loss >= alpha:
        return 1.0
    hoeffding = float(np.exp(-2 * n * (alpha - mean_loss) ** 2))
    bentkus = float(np.e * stats.binom.cdf(np.ceil(n * mean_loss), n, alpha))
    return min(1.0, hoeffding, bentkus)


@dataclass(frozen=True)
class Certificate:
    method: str
    lam: float  # inf means "never act"
    alpha: float
    delta: float | None
    n_runs: int
    empirical_risk: float  # calibration risk at the chosen lambda

    @property
    def vacuous(self) -> bool:
        return not np.isfinite(self.lam)


def _grid(scores: np.ndarray, size: int = 200) -> np.ndarray:
    """Candidate thresholds from most conservative (never act) to most permissive."""
    qs = np.unique(np.quantile(scores, np.linspace(0, 1, size)))
    return np.concatenate([[NEVER], qs[::-1]])


def crc_threshold(
    scores: np.ndarray, losses: np.ndarray, run_ids: Sequence[str], alpha: float
) -> Certificate:
    lambdas = _grid(scores)
    lr, runs = run_losses(scores, losses, run_ids, lambdas)
    n = len(runs)
    risk = lr.mean(axis=0)
    adjusted = n / (n + 1) * risk + 1 / (n + 1)
    # Thresholds run from "never act" to most permissive, so risk only grows along
    # the grid: scan until the first threshold that breaks the bound.
    j = 0
    for idx in range(1, len(lambdas)):
        if adjusted[idx] > alpha:
            break
        j = idx
    return Certificate("crc", float(lambdas[j]), alpha, None, n, float(risk[j]))


def ltt_threshold(
    scores: np.ndarray,
    losses: np.ndarray,
    run_ids: Sequence[str],
    alpha: float,
    delta: float,
) -> Certificate:
    lambdas = _grid(scores)
    lr, runs = run_losses(scores, losses, run_ids, lambdas)
    n = len(runs)
    risk = lr.mean(axis=0)
    chosen = 0  # "never act" is always safe
    for j in range(1, len(lambdas)):
        if hoeffding_bentkus_p(float(risk[j]), n, alpha) <= delta:
            chosen = j
        else:
            break
    return Certificate("ltt", float(lambdas[chosen]), alpha, delta, n, float(risk[chosen]))


def realised(
    scores: np.ndarray,
    losses: np.ndarray,
    rescues: np.ndarray,
    run_ids: Sequence[str],
    lam: float,
) -> dict[str, float]:
    """Held-out disruption risk, rescue rate and action rate at a threshold."""
    act = scores >= lam
    lr, _ = run_losses(scores, losses, run_ids, np.array([lam]))
    rr, _ = run_losses(scores, rescues, run_ids, np.array([lam]))
    return {
        "risk": float(lr.mean()),
        "rescue": float(rr.mean()),
        "act_rate": float(act.mean()),
    }
