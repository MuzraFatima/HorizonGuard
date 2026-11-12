import numpy as np
import pytest

from horizonguard import estimands as est

# Arm 0 = none, arm 1 = reminder. Columns share seeds.
Y = np.array(
    [
        [1, 1, 1, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 1, 0, 0, 0],
    ]
)


def test_arm_risk_and_wilson():
    risk = est.arm_risk(Y, 0)
    assert risk.value == pytest.approx(5 / 8)
    assert 0 <= risk.lower < risk.value < risk.upper <= 1


def test_benefit_equals_rescue_minus_disruption():
    dec = est.decompose(Y, 1)
    assert dec.rescue == pytest.approx(4 / 8)
    assert dec.disruption == pytest.approx(1 / 8)
    assert dec.rescue + dec.disruption + dec.both_fail + dec.both_succeed == pytest.approx(1)
    assert est.benefit(Y, 1).value == pytest.approx(dec.benefit)


def test_net_benefit_weights_disruption():
    dec = est.decompose(Y, 1)
    assert dec.net_benefit(2.0) == pytest.approx(4 / 8 - 2 / 8)
    interval = dec.net_benefit_estimate(2.0)
    assert interval.lower < interval.value < interval.upper


def test_best_arm_prefers_no_intervention_when_it_hurts():
    hurts = np.array([[0, 0, 0, 0], [1, 1, 0, 0]])
    assert est.best_arm(hurts) == (0, 0.0)
    arm, score = est.best_arm(Y)
    assert arm == 1 and score > 0


def test_recoverability_horizon_is_last_helpful_step():
    k = 200
    helpful = np.vstack([np.ones(k, dtype=int), np.zeros(k, dtype=int)])
    useless = np.vstack([np.ones(k, dtype=int), np.ones(k, dtype=int)])
    forks = {0: helpful, 5: helpful, 10: useless, 15: useless}
    assert est.recoverability_horizon(forks, eps=0.05) == 5
    assert est.recoverability_horizon({0: useless}, eps=0.05) is None
    assert est.fix_by_countdown(3, 5) == 2
    assert est.fix_by_countdown(3, None) is None


def test_lower_bound_makes_horizon_conservative():
    rng = np.random.default_rng(0)
    none = np.ones(10, dtype=int)
    weak = (rng.random(10) < 0.8).astype(int)  # small, noisy benefit
    forks = {4: np.vstack([none, weak])}
    assert est.recoverability_horizon(forks, eps=0.05, use_lower_bound=False) == 4
    assert est.recoverability_horizon(forks, eps=0.05, use_lower_bound=True) is None


@pytest.mark.parametrize("bad", [[1, 0, 1], [[0, 2], [1, 0]]])
def test_invalid_outcomes_rejected(bad):
    with pytest.raises(ValueError):
        est.arm_risk(np.array(bad))
