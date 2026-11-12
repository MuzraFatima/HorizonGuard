"""Lead-time scaling (RQ5, H5a).

Runs are generated on a grid of drift strength (a multiplier on every drift
driver) and context length (episode length, which also scales context
pressure). A monitor trained on DriftBench-CF scores every step with one fixed
threshold. Per cell we record the mean lead time over violating runs, then fit

    lead = a + b log(strength) + c log(context)

and report the out-of-sample R^2 from leave-one-cell-out prediction. An
exploratory fit adds ``(log strength)^2``. It is not preregistered and is
reported as hypothesis-generating only.

Quickest change detection gives a qualitative expectation: detection delay
shrinks roughly like log(1/alpha) / KL as the drift signal grows, while the
time to violation also shrinks. The fitted signs show which effect dominates.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from horizonguard.agents import make_agent
from horizonguard.bench.grid import DRIFT_TYPES
from horizonguard.envs import make_env
from horizonguard.eval.dataset import step_scores
from horizonguard.eval.metrics import lead_times
from horizonguard.fork import run_episode
from horizonguard.logging import StepRecord


def scaled_drivers(drift_type: str, strength: float):
    d = DRIFT_TYPES[drift_type]
    return replace(
        d,
        nudge_strength=d.nudge_strength * strength,
        distractor_load=d.distractor_load * strength,
        context_pressure=d.context_pressure * strength,
        goal_switch_strength=d.goal_switch_strength * strength,
    )


@dataclass
class Cell:
    strength: float
    context: int
    runs: int
    violation_rate: float
    detection_rate: float
    mean_lead: float


def run_cell(
    score: Callable[[Sequence[StepRecord]], float],
    threshold: float,
    strength: float,
    context: int,
    runs: int,
    envs: Sequence[str],
    agents: Sequence[str],
    drift_types: Sequence[str],
    seed: int,
    stride: int = 2,
) -> Cell:
    trajs = []
    for i in range(runs):
        env = make_env(
            envs[i % len(envs)],
            drivers=scaled_drivers(drift_types[i % len(drift_types)], strength),
            max_steps=context,
        )
        trajs.append(run_episode(env, make_agent(agents[i % len(agents)]), seed + i))
    scores = step_scores(trajs, score, stride=stride)
    viol = [t.violation_step for t in trajs]
    lt = lead_times(scores, viol, threshold)
    return Cell(
        strength,
        context,
        runs,
        float(np.mean([v is not None for v in viol])),
        float((lt > 0).mean()) if lt.size else float("nan"),
        float(lt.mean()) if lt.size else float("nan"),
    )


def _design(c: Cell, quadratic: bool) -> list[float]:
    ls = np.log(c.strength)
    row = [1.0, ls, np.log(c.context)]
    return [*row, ls**2] if quadratic else row


def fit_law(cells: Sequence[Cell], quadratic: bool = False) -> dict[str, Any]:
    """Least-squares law; ``quadratic`` adds (log strength)^2 (exploratory, not preregistered)."""
    ok = [c for c in cells if np.isfinite(c.mean_lead)]
    k = 4 if quadratic else 3
    if len(ok) < k + 1:
        return {"coef": [float("nan")] * k, "r2_in": float("nan"), "r2_loo": float("nan")}
    x = np.array([_design(c, quadratic) for c in ok])
    y = np.array([c.mean_lead for c in ok])
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    ss_tot = float(np.sum((y - y.mean()) ** 2)) or 1e-12
    r2_in = 1 - float(np.sum((y - x @ coef) ** 2)) / ss_tot
    preds = []
    for i in range(len(ok)):
        m = np.arange(len(ok)) != i
        c_i, *_ = np.linalg.lstsq(x[m], y[m], rcond=None)
        preds.append(float(x[i] @ c_i))
    r2_loo = 1 - float(np.sum((y - np.array(preds)) ** 2)) / ss_tot
    return {"coef": coef.tolist(), "r2_in": r2_in, "r2_loo": r2_loo}


def scaling_study(
    score: Callable[[Sequence[StepRecord]], float],
    threshold: float,
    strengths: Sequence[float] = (0.5, 1.0, 1.5, 2.0),
    contexts: Sequence[int] = (30, 45, 60),
    runs_per_cell: int = 24,
    envs: Sequence[str] = ("research", "sysadmin", "codebase"),
    agents: Sequence[str] = ("birch-react", "cobalt-react", "delta-plan", "atlas-react"),
    drift_types: Sequence[str] = ("nudge", "distractor", "context"),
    seed: int = 7_000,
) -> tuple[list[Cell], dict[str, Any]]:
    cells = [
        run_cell(score, threshold, s, c, runs_per_cell, envs, agents, drift_types, seed + 1000 * i)
        for i, (s, c) in enumerate((s, c) for s in strengths for c in contexts)
    ]
    return cells, {"preregistered": fit_law(cells), "exploratory": fit_law(cells, quadratic=True)}
