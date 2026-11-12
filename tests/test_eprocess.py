import numpy as np
import pytest

from horizonguard.guarantees.eprocess import BettingEProcess, detection, empirical_run_fpr


def test_wealth_is_nonnegative_and_alarm_threshold():
    proc = BettingEProcess(alpha=0.1, m0=0.3, kappa=1 / 0.3)
    w = proc.wealth(np.zeros(50))
    assert np.all(w >= 0)
    assert proc.first_alarm(np.zeros(50)) is None
    assert proc.first_alarm(np.ones(50)) is not None


def test_ville_bound_holds_under_the_null():
    rng = np.random.default_rng(0)
    healthy = [rng.beta(2, 8, size=200) for _ in range(400)]  # mean 0.2
    proc = BettingEProcess(alpha=0.05, m0=0.2)
    assert empirical_run_fpr(proc, healthy) <= 0.05


def test_calibration_and_detection():
    rng = np.random.default_rng(1)
    healthy = [rng.beta(2, 8, size=60) for _ in range(50)]
    proc = BettingEProcess.calibrate(healthy, alpha=0.05)
    assert proc.m0 > 0.2
    drifting = [np.concatenate([rng.beta(2, 8, 30), rng.beta(8, 2, 30)]) for _ in range(20)]
    out = detection(proc, drifting, [55] * 20)
    assert out["detection_rate"] > 0.9 and out["mean_lead"] > 5


def test_invalid_bets_rejected():
    with pytest.raises(ValueError):
        BettingEProcess(m0=0.5, kappa=3.0)
    with pytest.raises(ValueError):
        BettingEProcess(m0=0.0)


def test_plug_in_bets_are_predictable_and_bounded():
    proc = BettingEProcess(m0=0.25)
    x = np.array([0.9, 0.9, 0.1, 0.9])
    bets = proc.bets(x)
    assert bets[0] == 0.0
    assert np.all((bets >= 0) & (bets <= 0.9 / 0.25))
    # changing the current score must not change the current bet
    y = x.copy()
    y[3] = 0.0
    assert proc.bets(y)[3] == bets[3]
