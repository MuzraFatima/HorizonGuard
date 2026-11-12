import numpy as np
import pytest

from horizonguard.effects.policy import (
    EffectAwarePolicy,
    fix_by_countdown,
    horizon_errors,
    measured_horizons,
    predicted_horizons,
)
from horizonguard.eval.leaderboard import load_bench


@pytest.fixture(scope="module")
def bench(small_bench):
    return load_bench(small_bench)


def test_policy_picks_valid_arms_and_respects_margin(bench):
    policy = EffectAwarePolicy().fit(bench.part("train"))
    test = bench.part("test")
    arms = policy.decide(test)
    assert set(np.unique(arms)) <= {0, 1, 2, 3, 4}
    shy = EffectAwarePolicy(margin=10.0).fit(bench.part("train"))
    assert (shy.decide(test) == 0).all()
    p = policy.abstain_probability(test)
    assert np.all((p >= 0) & (p <= 1))


def test_measured_horizons_against_themselves(bench):
    data = bench.part("test")
    measured = measured_horizons(data, 0.05)
    errors = horizon_errors(measured, measured)
    assert errors["recoverable_agreement"] == 1.0
    assert errors["mean_abs_error"] in (0.0,) or np.isnan(errors["mean_abs_error"])
    pred = predicted_horizons(data, np.zeros((len(data), 4)), 0.05)
    assert all(v is None for v in pred.values())


def test_fix_by_countdown():
    assert fix_by_countdown([5, 10, 15], [0.5, 0.4, 0.3], eps=0.1) == pytest.approx(10.0)
    assert fix_by_countdown([5, 10], [0.2, 0.3], eps=0.1) is None
    assert fix_by_countdown([5, 10], [0.2, 0.05], eps=0.1) == 0.0
    assert fix_by_countdown([5], [0.3], eps=0.1) is None
