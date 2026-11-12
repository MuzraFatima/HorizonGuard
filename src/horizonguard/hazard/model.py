"""Multi-state hazard model of goal drift (Pillar 1).

The goal process has states Aligned (A), Drifting (D) and Violated (V, absorbing).
From the causal prefix features ``x_t`` the model predicts:

* a **state head** ``p(S_t = D | x_t)``. The state is latent at run time, so
  the monitor keeps a belief instead of a label;
* a **transition head from A**: stay, onset (A to D), abrupt violation (A to V);
* a **transition head from D**: recovery (D to A), stay, violation (D to V);
* an auxiliary **fidelity head** (ridge), used for diagnostics.

The heads are discrete-time competing-risk hazards fitted by multinomial
likelihood on observed transitions. Censored steps add no transition term,
which is the discrete-time right-censoring likelihood. Risk over a horizon
``H`` composes the per-step transition matrix under the current features:

    risk_H = (belief @ P(x_t)^H)[V]

A **risk head** (logistic, soft targets) reads the features together with the
hazard trunk (multi-horizon composed risk, drifting belief, onset and
violation hazards). It is trained on every step with its realised outcome and
on forked prefixes with their **rollout-grounded** risk, weighted higher
because fork labels are less noisy. That is the soft supervision that ties the
model to counterfactual ground truth rather than to realised outcomes alone.

The causal encoder here is the hand-built ``PrefixFeaturizer`` (smoothed
embeddings and goal geometry). A learned sequence encoder can replace it behind
the same features interface. That swap is logged as a deviation in the Phase 6
report.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from horizonguard.baselines.monitors import Monitor
from horizonguard.embeddings import PrefixFeaturizer
from horizonguard.eval.dataset import PrefixDataset
from horizonguard.hazard.data import StepData, build_step_data
from horizonguard.logging import StepRecord, Trajectory
from horizonguard.models import LogisticRegression, RidgeRegression
from horizonguard.models.softmax import SoftmaxRegression
from horizonguard.spec import State

_STEP_CACHE: dict[tuple[str, ...], StepData] = {}


def step_data_for(train: PrefixDataset, runs: Mapping[str, Trajectory]) -> StepData:
    """Step data for the runs in ``train``; cached because featurising every step is slow."""
    ids = tuple(sorted(set(train.meta["run_id"])))
    if ids not in _STEP_CACHE:
        _STEP_CACHE[ids] = build_step_data([runs[i] for i in ids if i in runs])
    return _STEP_CACHE[ids]


@dataclass
class HazardOutput:
    p_drifting: np.ndarray  # (n,)
    from_aligned: np.ndarray  # (n, 3) stay, onset, abrupt
    from_drifting: np.ndarray  # (n, 3) recovery, stay, violation

    def matrices(self) -> np.ndarray:
        n = len(self.p_drifting)
        p = np.zeros((n, 3, 3))
        p[:, 0, 0], p[:, 0, 1], p[:, 0, 2] = self.from_aligned.T
        p[:, 1, 0], p[:, 1, 1], p[:, 1, 2] = self.from_drifting.T
        p[:, 2, 2] = 1.0
        return p

    def belief(self) -> np.ndarray:
        return np.column_stack(
            [1 - self.p_drifting, self.p_drifting, np.zeros_like(self.p_drifting)]
        )


@dataclass
class MultiStateHazard(Monitor):
    name: str = "multi-state-hazard"
    horizon: int = 20
    l2: float = 3.0
    rollout_supervision: bool = True  # add fork-risk prefix rows to the risk head
    rollout_weight: float = 3.0
    groups: tuple[str, ...] | None = None  # restrict input channels (ablations)
    featurizer: PrefixFeaturizer = field(default_factory=PrefixFeaturizer)
    state_head: SoftmaxRegression = field(default_factory=lambda: SoftmaxRegression(2))
    head_a: SoftmaxRegression = field(default_factory=lambda: SoftmaxRegression(3))
    head_d: SoftmaxRegression = field(default_factory=lambda: SoftmaxRegression(3))
    fidelity_head: RidgeRegression = field(default_factory=RidgeRegression)
    risk_head: LogisticRegression = field(default_factory=LogisticRegression)
    _cols: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))

    # ----- fitting --------------------------------------------------------------------
    def _select(self, names: Sequence[str]) -> np.ndarray:
        if self.groups is None:
            return np.arange(len(names))
        return np.array([i for i, n in enumerate(names) if n.startswith(self.groups)])

    def fit_steps(self, steps: StepData) -> MultiStateHazard:
        self._cols = self._select(steps.feature_names)
        x = steps.x[:, self._cols]
        live = steps.state != State.VIOLATED
        self.state_head = SoftmaxRegression(2, l2=self.l2).fit(
            x[live], (steps.state[live] == State.DRIFTING).astype(int)
        )
        xa, ya = steps.transitions_from(State.ALIGNED)
        xd, yd = steps.transitions_from(State.DRIFTING)
        # from A: 0 stay, 1 onset, 2 abrupt; from D: 0 recovery, 1 stay, 2 violation
        self.head_a = SoftmaxRegression(3, l2=self.l2).fit(xa[:, self._cols], ya)
        if len(yd):
            self.head_d = SoftmaxRegression(3, l2=self.l2).fit(xd[:, self._cols], yd)
        else:  # no drifting steps seen: fall back to a sticky drifting state
            self.head_d = SoftmaxRegression(3, l2=self.l2).fit(
                np.zeros((3, len(self._cols))), np.array([0, 1, 2])
            )
        if steps.fidelity is not None and np.isfinite(steps.fidelity).any():
            ok = np.isfinite(steps.fidelity)
            self.fidelity_head = RidgeRegression(l2=10.0).fit(x[ok], steps.fidelity[ok])
        return self

    def trunk(self, x: np.ndarray) -> np.ndarray:
        """Hazard-derived features: multi-horizon risk, drifting belief, onset hazard."""
        x = np.atleast_2d(x)
        out = self.hazards(x)
        eps = 1e-4
        multi = np.clip(self.multi_horizon(x), eps, 1 - eps)
        onset = np.clip(out.from_aligned[:, 1], eps, 1 - eps)
        drift = np.clip(out.p_drifting, eps, 1 - eps)
        violation = np.clip(out.from_drifting[:, 2], eps, 1 - eps)
        stacked = np.column_stack([multi, onset, drift, violation])
        return np.log(stacked / (1 - stacked))

    def _head_input(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return np.column_stack([x[:, self._cols], self.trunk(x)])

    def fit(self, train: PrefixDataset, runs: Mapping[str, Trajectory]) -> MultiStateHazard:
        steps = step_data_for(train, runs)
        self.fit_steps(steps)
        # Step rows: realised violation within the horizon (censored rows dropped).
        first_v = np.full(steps.run.max() + 1, -1)
        for i in np.flatnonzero(steps.state == State.VIOLATED):
            if first_v[steps.run[i]] < 0:
                first_v[steps.run[i]] = steps.t[i]
        fv = first_v[steps.run]
        live = steps.state != State.VIOLATED
        y_step = ((fv > 0) & (fv - steps.t <= self.horizon)).astype(float)
        xs, ys, ws = [self._head_input(steps.x[live])], [y_step[live]], [np.ones(live.sum())]
        # Prefix rows: rollout-grounded soft risk from forks.
        if self.rollout_supervision and len(train):
            xs.append(self._head_input(train.x))
            ys.append(train.risk)
            ws.append(np.full(len(train), self.rollout_weight))
        self.risk_head = LogisticRegression(l2=self.l2).fit(
            np.vstack(xs), np.concatenate(ys), np.concatenate(ws)
        )
        return self

    # ----- prediction -----------------------------------------------------------------
    def hazards(self, x: np.ndarray) -> HazardOutput:
        x = np.atleast_2d(x)[:, self._cols]
        return HazardOutput(
            self.state_head.predict_proba(x)[:, 1],
            self.head_a.predict_proba(x),
            self.head_d.predict_proba(x),
        )

    def state_distribution(self, x: np.ndarray, horizon: int) -> np.ndarray:
        """(n, 3) predicted state distribution ``horizon`` steps ahead."""
        out = self.hazards(x)
        belief, p = out.belief(), out.matrices()
        for _ in range(horizon):
            belief = np.einsum("ni,nij->nj", belief, p)
        return belief

    def raw_risk(self, x: np.ndarray, horizon: int | None = None) -> np.ndarray:
        return self.state_distribution(x, horizon or self.horizon)[:, State.VIOLATED]

    def onset_risk(self, x: np.ndarray, horizon: int | None = None) -> np.ndarray:
        """P(leaving Aligned within the horizon | currently aligned)."""
        a = self.hazards(x).from_aligned
        return 1 - a[:, 0] ** (horizon or self.horizon)

    def risk(self, x: np.ndarray, horizon: int | None = None) -> np.ndarray:
        """Risk of violation within the horizon from the supervised risk head.

        Other horizons fall back to the raw hazard composition.
        """
        if horizon not in (None, self.horizon):
            return self.raw_risk(x, horizon)
        return self.risk_head.predict_proba(self._head_input(x))

    def predict(self, data: PrefixDataset) -> np.ndarray:
        return self.risk(data.x)

    def score_steps(self, steps: Sequence[StepRecord]) -> float:
        return float(self.risk(self.featurizer(steps)[None, :])[0])

    def multi_horizon(self, x: np.ndarray, horizons: Sequence[int] = (5, 10, 20)) -> np.ndarray:
        return np.column_stack([self.raw_risk(x, h) for h in horizons])
