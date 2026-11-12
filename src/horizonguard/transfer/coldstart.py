"""Cold start: how much data from a new agent restores the guarantee.

For an unseen value of an axis (for example a new model family), take ``N``
runs from it and recalibrate the disruption certificate four ways:

* ``source``: certificate from seen agents only (no target data);
* ``target``: target runs only. This needs ``N >= 1/alpha - 1`` before it can act;
* ``pooled``: seen calibration runs plus the ``N`` target runs;
* ``weighted``: seen calibration runs reweighted by an estimated density ratio
  ``p(target | x) / p(source | x)`` (covariate-shift weighted risk control), with
  the ``N`` target runs used to fit the ratio.

Each is scored on the *remaining* target runs, over random draws of which
target runs are used. The effect model is adapted few-shot by refitting with
the ``N`` target runs upweighted.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from horizonguard.effects.learners import DecompositionLearner
from horizonguard.eval.dataset import PrefixDataset
from horizonguard.guarantees.policy import CertifiedPolicy
from horizonguard.guarantees.risk_control import NEVER, crc_threshold, realised, run_losses
from horizonguard.models import LogisticRegression

METHODS = ("source", "target", "pooled", "weighted")


def _runs(data: PrefixDataset) -> list[str]:
    return sorted(set(data.meta["run_id"]))


def weighted_crc(
    scores: np.ndarray,
    losses: np.ndarray,
    run_ids: Sequence[str],
    run_weights: dict[str, float],
    alpha: float,
) -> float:
    """Weighted CRC threshold: weighted run risk plus the largest-weight correction."""
    lambdas = np.concatenate(
        [[NEVER], np.unique(np.quantile(scores, np.linspace(0, 1, 200)))[::-1]]
    )
    lr, runs = run_losses(scores, losses, run_ids, lambdas)
    w = np.array([run_weights[r] for r in runs])
    w_test = float(w.max())  # conservative stand-in for the unknown test weight
    total = w.sum() + w_test
    adjusted = (w[:, None] * lr).sum(axis=0) / total + w_test / total
    j = 0
    for idx in range(1, len(lambdas)):
        if adjusted[idx] > alpha:
            break
        j = idx
    return float(lambdas[j])


def density_ratio(source: PrefixDataset, target: PrefixDataset) -> LogisticRegression:
    x = np.vstack([source.x, target.x])
    y = np.concatenate([np.zeros(len(source)), np.ones(len(target))])
    return LogisticRegression(l2=10.0).fit(x, y)


def cold_start_curve(
    data: PrefixDataset,
    axis: str,
    value: str,
    sizes: Sequence[int] = (0, 5, 10, 20),
    alpha: float = 0.02,
    reps: int = 30,
    seed: int = 0,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    is_target = np.array([r[axis] == value for r in data.records])
    source, target = data.subset(~is_target), data.subset(is_target)
    src_runs = _runs(source)
    perm = list(rng.permutation(src_runs))
    cal_n = max(1, len(perm) // 4)
    src_cal, src_train = source.select_runs(perm[:cal_n]), source.select_runs(perm[cal_n:])
    base = CertifiedPolicy(method="crc", alpha=alpha).fit(src_train)
    tgt_runs = _runs(target)
    rows = []
    for n in sizes:
        if n >= len(tgt_runs) - 2:
            continue
        acc: dict[str, list[tuple[float, float, bool]]] = {m: [] for m in METHODS}
        eff_err: list[float] = []
        for _ in range(reps):
            order = list(rng.permutation(tgt_runs))
            few, rest = target.select_runs(order[:n]), target.select_runs(order[n:])
            s_rest = base.scores(rest)
            score_rest = np.where(np.isfinite(s_rest.score), s_rest.score, -1e9)
            for method in METHODS:
                lam = _threshold(method, base, src_cal, few, alpha, n)
                out = realised(score_rest, s_rest.loss, s_rest.rescue, list(s_rest.run_ids), lam)
                acc[method].append((out["risk"], out["rescue"], not np.isfinite(lam)))
            eff_err.append(_few_shot_effect_error(src_train, few, rest, n))
        row: dict[str, Any] = {"n_target": n, "effect_mae": float(np.mean(eff_err))}
        for method, vals in acc.items():
            arr = np.array(vals, dtype=float)
            row[f"{method}_violation"] = float(np.mean(arr[:, 0] > alpha))
            row[f"{method}_rescue"] = float(arr[:, 1].mean())
            row[f"{method}_vacuous"] = float(arr[:, 2].mean())
        rows.append(row)
    return rows


def _threshold(
    method: str,
    base: CertifiedPolicy,
    src_cal: PrefixDataset,
    few: PrefixDataset,
    alpha: float,
    n: int,
) -> float:
    def scored(d: PrefixDataset):
        s = base.scores(d)
        return np.where(np.isfinite(s.score), s.score, -1e9), s.loss, list(s.run_ids)

    if method == "source" or (n == 0 and method != "target"):
        sc, lo, ids = scored(src_cal)
        return max(crc_threshold(sc, lo, ids, alpha).lam, 0.0)
    if method == "target":
        if n == 0:
            return NEVER
        sc, lo, ids = scored(few)
        return max(crc_threshold(sc, lo, ids, alpha).lam, 0.0)
    if method == "pooled":
        sc1, lo1, id1 = scored(src_cal)
        sc2, lo2, id2 = scored(few)
        lam = crc_threshold(
            np.concatenate([sc1, sc2]), np.concatenate([lo1, lo2]), id1 + id2, alpha
        ).lam
        return max(lam, 0.0)
    ratio = density_ratio(src_cal, few)
    p = np.clip(ratio.predict_proba(src_cal.x), 1e-3, 1 - 1e-3)
    w_prefix = p / (1 - p)
    run_ids = src_cal.meta["run_id"]
    weights = {r: float(w_prefix[run_ids == r].mean()) for r in np.unique(run_ids)}
    sc, lo, ids = scored(src_cal)
    return max(weighted_crc(sc, lo, ids, weights, alpha), 0.0)


def _few_shot_effect_error(
    src_train: PrefixDataset, few: PrefixDataset, rest: PrefixDataset, n: int, boost: float = 5.0
) -> float:
    if n == 0:
        model = DecompositionLearner().fit(src_train)
    else:
        both = PrefixDataset(
            np.vstack([src_train.x] + [few.x] * int(boost)),
            src_train.feature_names,
            src_train.records + few.records * int(boost),
            np.concatenate([src_train.latent] + [few.latent] * int(boost)),
            {
                k: np.concatenate([src_train.meta[k]] + [few.meta[k]] * int(boost))
                for k in src_train.meta
            },
        )
        model = DecompositionLearner().fit(both)
    pred = model.predict_effects(rest)
    return float(np.mean(np.abs(pred - rest.arm("benefit"))))
