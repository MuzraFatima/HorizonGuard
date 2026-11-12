import numpy as np
import pytest

from horizonguard.effects.learners import (
    LEARNERS,
    DecompositionLearner,
    TLearner,
    constant_effect_baseline,
    cross_fit,
    effect_calibration,
    run_folds,
)
from horizonguard.eval.leaderboard import load_bench


@pytest.fixture(scope="module")
def bench(small_bench):
    return load_bench(small_bench)


@pytest.mark.parametrize("learner", LEARNERS, ids=lambda c: c.__name__)
def test_learners_predict_one_effect_per_arm(bench, learner):
    model = learner().fit(bench.part("train"))
    p = model.predict_effects(bench.part("test"))
    assert p.shape == (len(bench.part("test")), 4)
    assert np.all(np.isfinite(p))


def test_decomposition_is_consistent(bench):
    model = DecompositionLearner().fit(bench.part("train"))
    test = bench.part("test")
    r, d = model.predict_rescue_disruption(test)
    assert np.all((r >= 0) & (r <= 1) & (d >= 0) & (d <= 1))
    np.testing.assert_allclose(model.predict_net(test, 1.0), model.predict_effects(test))
    assert np.all(model.predict_net(test, 2.0) <= model.predict_effects(test) + 1e-12)


def test_t_learner_risks_are_probabilities(bench):
    risks = TLearner().fit(bench.part("train")).predict_risks(bench.part("test"))
    assert risks.shape[1] == 5 and np.all((risks >= 0) & (risks <= 1))


def test_cross_fitting_keeps_runs_together(bench):
    data = bench.part("train")
    folds = run_folds(data, 3)
    for run in np.unique(data.meta["run_id"]):
        assert len(set(folds[data.meta["run_id"] == run])) == 1
    oof = cross_fit(DecompositionLearner, data, folds=3)
    assert oof.shape == (len(data), 4)


def test_effect_calibration_and_constant_baseline(bench):
    train, test = bench.part("train"), bench.part("test")
    const = constant_effect_baseline(train, test)
    assert const.shape == (len(test), 4)
    perfect = effect_calibration(test.arm("benefit"), test.arm("benefit"))
    assert perfect["mae"] == 0 and perfect["slope"] == pytest.approx(1.0)
