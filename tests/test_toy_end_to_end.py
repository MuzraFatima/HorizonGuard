"""Phase 2 exit gate: every estimand computed end to end on the toy environment."""

import numpy as np

from horizonguard import estimands as est
from horizonguard.envs import ToyDriftProcess, ToyParams, continue_from, fork_outcomes
from horizonguard.labels import onset_interval
from horizonguard.spec import Intervention, State


def forced_drift(**overrides):
    """A process that starts drifting at step 0 and never recovers on its own."""
    params = ToyParams(onset_rate=0.0, recovery_rate=0.0, **overrides)
    proc = ToyDriftProcess(params)
    proc.state.drifting = True
    return proc, params


def drifting_run(seed=0, steps=6):
    """A run that is drifting but not yet violated after ``steps`` steps."""
    proc, params = forced_drift()
    for _ in range(steps):
        proc.step(seed)
    assert proc.state.state is State.DRIFTING
    return proc, params


def test_snapshot_restore_replays_identically():
    proc = ToyDriftProcess()
    proc.run(10, seed=3)
    snap = proc.snapshot()
    first = [proc.step(seed=3) for _ in range(15)]
    proc.restore(snap)
    second = [proc.step(seed=3) for _ in range(15)]
    assert first == second


def test_fork_from_prefix_matches_unforked_distribution():
    params = ToyParams()
    horizon, n = 20, 400
    unforked = []
    proc = ToyDriftProcess(params)
    proc.run(5, seed=0)
    snap = proc.snapshot()
    for seed in range(n):
        p = ToyDriftProcess(params)
        p.restore(snap)
        unforked.append(int(State.VIOLATED in p.run(horizon, seed=seed)))
    forked = fork_outcomes(snap, (Intervention.NONE,), k=n, horizon=horizon, params=params)
    assert np.mean(unforked) == np.mean(forked[0])


def test_all_estimands_on_a_drifting_prefix():
    proc, params = drifting_run()
    y = fork_outcomes(proc.snapshot(), k=200, horizon=20, params=params)
    risk = est.arm_risk(y, 0)
    assert risk.value > 0.5  # left alone, a drifting toy agent usually violates
    dec = est.decompose(y, Intervention.RESTATEMENT)
    assert dec.rescue > dec.disruption
    assert est.benefit(y, Intervention.RESTATEMENT).excludes_zero
    assert dec.net_benefit(2.0) <= dec.benefit


def test_recoverability_horizon_exists_and_closes():
    proc, params = forced_drift()
    forks = {}
    for t in range(14):
        if proc.state.violated:
            break
        forks[t] = fork_outcomes(proc.snapshot(), k=100, horizon=20, params=params)
        proc.step(seed=1)
    horizon = est.recoverability_horizon(forks, eps=0.05)
    assert horizon is not None
    assert horizon < max(forks)  # intervening late stops helping
    assert est.fix_by_countdown(0, horizon) > 0


def test_intervention_on_healthy_prefix_can_only_disrupt():
    proc = ToyDriftProcess(ToyParams(onset_rate=0.0))
    proc.run(5, seed=0)
    y = fork_outcomes(proc.snapshot(), k=300, horizon=20, params=proc.params)
    dec = est.decompose(y, Intervention.ESCALATION)
    assert dec.rescue == 0.0
    assert dec.disruption >= 0.0


def test_onset_interval_covers_true_onset_in_toy_run():
    params = ToyParams(onset_rate=0.0, recovery_rate=0.0, limit=3.0)
    proc = ToyDriftProcess(params)
    proc.run(20, seed=4)
    proc.state.drifting = True
    proc.run(15, seed=4)
    interval = onset_interval(np.array(proc.state.history), model="hinge")
    assert interval is not None and interval.lower <= 21 and interval.upper >= 19


def test_continuation_from_violated_snapshot_is_violated():
    proc, _ = forced_drift(drift_speed=0.5)
    proc.run(5, seed=0)
    assert proc.state.violated
    assert continue_from(proc.snapshot(), Intervention.GATE, 0, 5) == 1
