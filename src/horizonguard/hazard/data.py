"""Step-level data for the multi-state hazard model.

Every step of every training run becomes a row: causal prefix features, the
current state and the next state. Runs that end without violation are
right-censored. Their last row simply has no further transition, which the
discrete-time likelihood handles by using only observed transitions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from horizonguard.bench.generate import run_labels
from horizonguard.embeddings import PrefixFeaturizer
from horizonguard.logging import Trajectory
from horizonguard.spec import State, states_from_labels


@dataclass
class StepData:
    x: np.ndarray  # (n, d) features of the prefix ending at step t
    state: np.ndarray  # (n,) state at t
    next_state: np.ndarray  # (n,) state at t + 1, -1 when censored
    run: np.ndarray  # (n,) run index
    t: np.ndarray  # (n,)
    agent: np.ndarray  # (n,)
    feature_names: list[str]

    def __len__(self) -> int:
        return len(self.state)

    def transitions_from(self, source: State) -> tuple[np.ndarray, np.ndarray]:
        """Rows leaving ``source`` with an observed next state."""
        mask = (self.state == source) & (self.next_state >= 0)
        return self.x[mask], self.next_state[mask]


def run_states(traj: Trajectory, min_drop: float = 0.15) -> list[State]:
    onset, _ = run_labels(traj, min_drop)
    return states_from_labels(onset, traj.violation_step, len(traj))


def build_step_data(
    runs: Sequence[Trajectory],
    featurizer: PrefixFeaturizer | None = None,
    min_drop: float = 0.15,
) -> StepData:
    featurizer = featurizer or PrefixFeaturizer()
    xs, states, nexts, run_idx, ts, agents = [], [], [], [], [], []
    for i, traj in enumerate(runs):
        labels = run_states(traj, min_drop)
        for t in range(len(traj)):
            xs.append(featurizer(traj.steps[: t + 1]))
            states.append(int(labels[t]))
            nexts.append(int(labels[t + 1]) if t + 1 < len(traj) else -1)
            run_idx.append(i)
            ts.append(t + 1)
            agents.append(traj.steps[0].agent)
    return StepData(
        np.array(xs),
        np.array(states),
        np.array(nexts),
        np.array(run_idx),
        np.array(ts),
        np.array(agents),
        featurizer.feature_names(),
    )
