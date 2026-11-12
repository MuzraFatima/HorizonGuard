"""A toy goal-drift process for checking every estimand end to end.

The latent drift level ``d`` starts at zero. Each step, an aligned agent starts
drifting with probability ``onset_rate``. A drifting agent accumulates drift
and recovers on its own with probability ``recovery_rate``. The run is violated
once ``d`` reaches ``limit``. Observed fidelity is ``1 - d / limit`` plus noise.

An intervention of strength ``s`` is complied with with a probability that
falls as drift deepens, which is what produces a recoverability horizon. When
complied with, it ends the drift episode and removes a share of the drift.
Independently, it disrupts the agent with probability ``disruption[s]``,
adding a jump of drift. That is the harm the guarantees later control.

Randomness is drawn from streams keyed by ``(seed, step, purpose)``, so arms
that share a seed see the same environment noise. This is the common random
numbers design from the specification.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from horizonguard.spec import Intervention, State

_ONSET, _DRIFT, _RECOVER, _OBS, _COMPLY, _DISRUPT = range(6)


@dataclass(frozen=True)
class ToyParams:
    onset_rate: float = 0.04
    drift_speed: float = 0.08
    drift_noise: float = 0.03
    recovery_rate: float = 0.03
    limit: float = 1.0
    obs_noise: float = 0.03
    reduction: tuple[float, ...] = (0.0, 0.35, 0.55, 0.7, 0.9)
    disruption: tuple[float, ...] = (0.0, 0.03, 0.06, 0.1, 0.15)
    disruption_jump: float = 0.45
    comply_slope: float = 10.0  # how fast compliance falls with drift
    comply_mid: float = 0.5  # drift share at which compliance is 50%


@dataclass
class ToyState:
    t: int = 0
    drift: float = 0.0
    drifting: bool = False
    violated: bool = False
    history: list[float] = field(default_factory=list)  # observed fidelity

    @property
    def state(self) -> State:
        if self.violated:
            return State.VIOLATED
        return State.DRIFTING if self.drifting else State.ALIGNED


def _rng(seed: int, t: int, purpose: int) -> np.random.Generator:
    return np.random.default_rng([seed, t, purpose])


class ToyDriftProcess:
    def __init__(self, params: ToyParams | None = None, state: ToyState | None = None):
        self.params = params or ToyParams()
        self.state = state or ToyState()

    def snapshot(self) -> ToyState:
        return copy.deepcopy(self.state)

    def restore(self, snap: ToyState) -> None:
        self.state = copy.deepcopy(snap)

    def intervene(self, strength: Intervention, seed: int) -> None:
        if strength is Intervention.NONE or self.state.violated:
            return
        p, st = self.params, self.state
        share = st.drift / p.limit
        p_comply = 1.0 / (1.0 + np.exp(p.comply_slope * (share - p.comply_mid)))
        if _rng(seed, st.t, _COMPLY).random() < p_comply:
            # A complied re-anchoring stops the drift episode and removes part of
            # the accumulated drift; a fresh onset can still happen later.
            st.drift *= 1.0 - p.reduction[strength]
            st.drifting = False
        if _rng(seed, st.t, _DISRUPT).random() < p.disruption[strength]:
            st.drift += p.disruption_jump * p.limit
            st.drifting = True

    def step(self, seed: int) -> float:
        p, st = self.params, self.state
        if not st.violated:
            if not st.drifting and _rng(seed, st.t, _ONSET).random() < p.onset_rate:
                st.drifting = True
            if st.drifting:
                if _rng(seed, st.t, _RECOVER).random() < p.recovery_rate:
                    st.drifting = False
                else:
                    noise = _rng(seed, st.t, _DRIFT).normal(0, p.drift_noise)
                    st.drift = max(0.0, st.drift + p.drift_speed + noise)
            else:
                st.drift = max(0.0, st.drift * 0.8)
            if st.drift >= p.limit:
                st.violated = True
        fid = 0.0 if st.violated else 1.0 - st.drift / p.limit
        fid = float(np.clip(fid + _rng(seed, st.t, _OBS).normal(0, p.obs_noise), 0.0, 1.0))
        st.history.append(fid)
        st.t += 1
        return fid

    def run(self, steps: int, seed: int) -> list[State]:
        states = []
        for _ in range(steps):
            self.step(seed)
            states.append(self.state.state)
        return states


def continue_from(
    snap: ToyState,
    strength: Intervention,
    seed: int,
    horizon: int,
    params: ToyParams | None = None,
) -> int:
    """One continuation: apply ``strength`` at the snapshot, run ``horizon`` steps.

    Returns Y = 1 if a violation occurs within the horizon.
    """
    proc = ToyDriftProcess(params, copy.deepcopy(snap))
    if proc.state.violated:
        return 1
    proc.intervene(strength, seed)
    for _ in range(horizon):
        proc.step(seed)
        if proc.state.violated:
            return 1
    return 0


def fork_outcomes(
    snap: ToyState,
    strengths: tuple[Intervention, ...] = tuple(Intervention),
    k: int = 32,
    horizon: int = 20,
    base_seed: int = 0,
    params: ToyParams | None = None,
) -> np.ndarray:
    """Outcome matrix of shape (len(strengths), k); column j uses seed base_seed + j."""
    if strengths[0] is not Intervention.NONE:
        raise ValueError("the first arm must be the no-intervention arm")
    return np.array(
        [
            [continue_from(snap, s, base_seed + j, horizon, params) for j in range(k)]
            for s in strengths
        ],
        dtype=np.int8,
    )
