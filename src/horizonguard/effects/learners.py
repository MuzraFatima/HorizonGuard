"""Per-prefix intervention-effect estimators (Pillar 2).

Fork labels observe every arm on the same seeds, so the effect of arm ``s`` at a
prefix has an unbiased (if noisy) paired label. The learners differ in what
they model:

* ``TLearner``: separate soft-label risk models per arm, effect = mu_0 - mu_s;
* ``DecompositionLearner``: rescue and disruption probabilities per arm, so
  net benefit ``r - w d`` can be formed for any preregistered weight ``w``
  (the main model);
* ``DirectEffectLearner``: ridge on the paired benefit, weighted by the
  inverse of each label's noise plus between-prefix variance (shrinkage);
* ``DRLearner``: the deployment-realistic case, where only one randomly
  assigned arm per prefix is observed. Doubly robust pseudo-outcomes stand in
  for the paired difference. It shows what forks buy over observational logs.

All expose ``predict_effects(x) -> (n, n_arms)`` (benefit per active arm) and,
where available, ``predict_rescue_disruption``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from horizonguard.eval.dataset import PrefixDataset
from horizonguard.models import LogisticRegression, RidgeRegression

N_ARMS = 4


def _features(data: PrefixDataset, extra: Callable[[np.ndarray], np.ndarray] | None) -> np.ndarray:
    return data.x if extra is None else np.column_stack([data.x, extra(data.x)])


@dataclass
class TLearner:
    name: str = "t-learner"
    l2: float = 3.0
    extra: Callable[[np.ndarray], np.ndarray] | None = None
    models: list[LogisticRegression] = field(default_factory=list)

    def fit(self, data: PrefixDataset) -> TLearner:
        x = _features(data, self.extra)
        y = np.array([r["outcomes"] for r in data.records], dtype=float).mean(axis=2)  # (n, 5)
        self.models = [LogisticRegression(l2=self.l2).fit(x, y[:, a]) for a in range(N_ARMS + 1)]
        return self

    def predict_risks(self, data: PrefixDataset) -> np.ndarray:
        x = _features(data, self.extra)
        return np.column_stack([m.predict_proba(x) for m in self.models])

    def predict_effects(self, data: PrefixDataset) -> np.ndarray:
        risks = self.predict_risks(data)
        return risks[:, :1] - risks[:, 1:]


@dataclass
class DecompositionLearner:
    name: str = "decomposition"
    l2: float = 3.0
    extra: Callable[[np.ndarray], np.ndarray] | None = None
    rescue: list[LogisticRegression] = field(default_factory=list)
    disruption: list[LogisticRegression] = field(default_factory=list)

    def fit(self, data: PrefixDataset) -> DecompositionLearner:
        x = _features(data, self.extra)
        r, d = data.arm("rescue"), data.arm("disruption")
        self.rescue = [LogisticRegression(l2=self.l2).fit(x, r[:, a]) for a in range(N_ARMS)]
        self.disruption = [LogisticRegression(l2=self.l2).fit(x, d[:, a]) for a in range(N_ARMS)]
        return self

    def predict_rescue_disruption(self, data: PrefixDataset) -> tuple[np.ndarray, np.ndarray]:
        x = _features(data, self.extra)
        r = np.column_stack([m.predict_proba(x) for m in self.rescue])
        d = np.column_stack([m.predict_proba(x) for m in self.disruption])
        return r, d

    def predict_effects(self, data: PrefixDataset) -> np.ndarray:
        r, d = self.predict_rescue_disruption(data)
        return r - d

    def predict_net(self, data: PrefixDataset, weight: float) -> np.ndarray:
        r, d = self.predict_rescue_disruption(data)
        return r - weight * d


@dataclass
class DirectEffectLearner:
    name: str = "direct-shrunk"
    l2: float = 10.0
    extra: Callable[[np.ndarray], np.ndarray] | None = None
    models: list[RidgeRegression] = field(default_factory=list)

    def fit(self, data: PrefixDataset) -> DirectEffectLearner:
        from horizonguard.bench.quality import within_variance

        x = _features(data, self.extra)
        b = data.arm("benefit")
        self.models = []
        for a in range(N_ARMS):
            noise = np.array([within_variance(r, a) for r in data.records])
            tau2 = max(float(np.var(b[:, a]) - noise.mean()), 1e-4)
            w = 1.0 / (noise + tau2)
            self.models.append(RidgeRegression(l2=self.l2).fit(x, b[:, a], weight=w / w.mean()))
        return self

    def predict_effects(self, data: PrefixDataset) -> np.ndarray:
        x = _features(data, self.extra)
        return np.column_stack([m.predict(x) for m in self.models])


@dataclass
class DRLearner:
    """Doubly robust effects from one randomly assigned arm per prefix.

    Each prefix keeps the outcome of a single continuation under arm 0 (with
    probability ``1 - propensity``) or under arm ``s`` (with ``propensity``),
    the way a logged deployment with randomised interventions would look.
    """

    name: str = "dr-observational"
    propensity: float = 0.5
    l2: float = 10.0
    seed: int = 0
    models: list[RidgeRegression] = field(default_factory=list)

    def fit(self, data: PrefixDataset) -> DRLearner:
        rng = np.random.default_rng(self.seed)
        y = np.array([r["outcomes"] for r in data.records], dtype=float)  # (n, 5, k)
        n, _, k = y.shape
        col = rng.integers(0, k, n)
        x = data.x
        self.models = []
        for a in range(1, N_ARMS + 1):
            treated = rng.random(n) < self.propensity
            obs = np.where(treated, y[np.arange(n), a, col], y[np.arange(n), 0, col])
            mu0 = LogisticRegression(l2=3.0).fit(x[~treated], obs[~treated]).predict_proba(x)
            mu1 = LogisticRegression(l2=3.0).fit(x[treated], obs[treated]).predict_proba(x)
            e = self.propensity
            # benefit = risk(none) - risk(arm): sign flipped relative to the usual ATE
            pseudo = (mu0 - mu1) + np.where(treated, -(obs - mu1) / e, (obs - mu0) / (1 - e))
            self.models.append(RidgeRegression(l2=self.l2).fit(x, pseudo))
        return self

    def predict_effects(self, data: PrefixDataset) -> np.ndarray:
        return np.column_stack([m.predict(data.x) for m in self.models])


def run_folds(data: PrefixDataset, folds: int, seed: int = 0) -> np.ndarray:
    """Fold index per prefix, assigned by run so no run straddles folds."""
    runs = np.unique(data.meta["run_id"])
    rng = np.random.default_rng(seed)
    assignment = dict(zip(runs, rng.permutation(len(runs)) % folds, strict=True))
    return np.array([assignment[r] for r in data.meta["run_id"]])


def cross_fit(
    make: Callable[[], object], data: PrefixDataset, folds: int = 5, seed: int = 0
) -> np.ndarray:
    """Out-of-fold effect predictions, (n, n_arms)."""
    fold = run_folds(data, folds, seed)
    out = np.zeros((len(data), N_ARMS))
    for f in range(folds):
        test = fold == f
        if not test.any() or test.all():
            continue
        model = make().fit(data.subset(~test))
        out[test] = model.predict_effects(data.subset(test))
    return out


def effect_calibration(pred: np.ndarray, label: np.ndarray, bins: int = 5) -> dict[str, float]:
    """Calibration of effects, not just rankings: slope and binned error vs paired labels."""
    pred, label = np.ravel(pred), np.ravel(label)
    slope = float(np.cov(pred, label)[0, 1] / max(np.var(pred), 1e-12))
    order = np.argsort(pred)
    chunks = np.array_split(order, bins)
    binned = float(np.mean([abs(pred[c].mean() - label[c].mean()) for c in chunks if len(c)]))
    return {
        "corr": float(np.corrcoef(pred, label)[0, 1]) if pred.std() > 0 else float("nan"),
        "slope": slope,
        "binned_error": binned,
        "mae": float(np.mean(np.abs(pred - label))),
    }


def constant_effect_baseline(train: PrefixDataset, test: PrefixDataset) -> np.ndarray:
    """Per-agent mean effect: the preregistered kill comparison for RQ2."""
    b = train.arm("benefit")
    agents_train, agents_test = train.meta["agent"], test.meta["agent"]
    pooled = b.mean(axis=0)
    out = np.empty((len(test), N_ARMS))
    for i, a in enumerate(agents_test):
        m = agents_train == a
        out[i] = b[m].mean(axis=0) if m.any() else pooled
    return out


LEARNERS: Sequence[Callable[[], object]] = (
    TLearner,
    DecompositionLearner,
    DirectEffectLearner,
    DRLearner,
)
