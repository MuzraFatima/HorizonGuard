import numpy as np
import pytest

from horizonguard.labels import (
    Check,
    Constraint,
    FidelityScorer,
    JudgeCalibrator,
    cusum_onset,
    isotonic_fit,
    krippendorff_alpha,
    onset_interval,
    pelt_changepoints,
    pelt_onset,
)


def drifting_series(onset=30, n=60, drop=0.4, noise=0.03, seed=0):
    rng = np.random.default_rng(seed)
    x = np.full(n, 0.9) + rng.normal(0, noise, n)
    x[onset:] -= drop
    return x


def test_isotonic_fit_is_monotone():
    knots, values = isotonic_fit([0.1, 0.4, 0.3, 0.9], [0, 1, 0, 1])
    assert np.all(np.diff(values) >= 0)
    assert len(knots) == len(values)


def test_judge_calibrator_maps_into_unit_interval():
    rng = np.random.default_rng(1)
    raw = rng.random(200)
    human = (rng.random(200) < raw**2).astype(float)
    cal = JudgeCalibrator.fit(raw, human)
    assert 0 <= cal(0.0) <= cal(0.5) <= cal(1.0) <= 1


def test_hard_violation_zeros_fidelity():
    budget = Constraint(
        "budget", lambda p: Check.VIOLATED if sum(p) > 10 else Check.SATISFIED, hard=True
    )
    style = Constraint("style", lambda p: Check.NOT_APPLICABLE, hard=False)
    scorer = FidelityScorer([budget, style])
    assert scorer.score([1, 2]) == 1.0
    assert scorer.score([5, 6]) == 0.0
    assert scorer.hard_violation([5, 6])
    assert scorer.trajectory([4, 4, 4]).tolist() == [1.0, 1.0, 0.0]


def test_judge_blends_with_constraints():
    ok = Constraint("ok", lambda p: Check.SATISFIED)
    scorer = FidelityScorer([ok], judge=lambda p: 0.2)
    assert scorer.score([0]) == pytest.approx(0.5 * 1.0 + 0.5 * 0.2)


def test_detectors_find_planted_onset():
    x = drifting_series(onset=30)
    assert abs(cusum_onset(x) - 30) <= 3
    assert abs(pelt_onset(x) - 30) <= 2
    interval = onset_interval(x)
    assert interval.contains(30)
    assert interval.drop == pytest.approx(0.4, abs=0.05)


def test_detectors_stay_quiet_on_healthy_runs():
    rng = np.random.default_rng(3)
    x = 0.9 + rng.normal(0, 0.03, 60)
    assert cusum_onset(x) is None
    assert pelt_changepoints(x) == []


def test_interval_widens_with_noise():
    narrow = onset_interval(drifting_series(noise=0.02, seed=4))
    wide = onset_interval(drifting_series(noise=0.15, seed=4))
    assert (wide.upper - wide.lower) >= (narrow.upper - narrow.lower)


def test_krippendorff_alpha_extremes():
    perfect = np.array([[1, 2, 3, 4], [1, 2, 3, 4]], dtype=float)
    assert krippendorff_alpha(perfect) == pytest.approx(1.0)
    rng = np.random.default_rng(5)
    noise = rng.integers(0, 2, size=(3, 300)).astype(float)
    assert abs(krippendorff_alpha(noise, level="nominal")) < 0.1


def test_krippendorff_alpha_reference_value():
    # Krippendorff (2011) worked example, nominal data with missing ratings.
    nan = np.nan
    data = np.array(
        [
            [1, 2, 3, 3, 2, 1, 4, 1, 2, nan, nan, nan],
            [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, nan, 3],
            [nan, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, nan],
            [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, nan],
        ]
    )
    assert krippendorff_alpha(data, level="nominal") == pytest.approx(0.743, abs=0.001)


def test_detector_accuracy_over_many_seeds():
    cusum_hits = pelt_hits = covered = false_alarms = 0
    seeds = range(60)
    for seed in seeds:
        healthy = 0.9 + np.random.default_rng(1000 + seed).normal(0, 0.03, 60)
        false_alarms += pelt_changepoints(healthy) != []
        x = drifting_series(onset=30, seed=seed)
        cusum_hits += abs(cusum_onset(x) - 30) <= 2
        pelt_hits += abs(pelt_onset(x) - 30) <= 2
        covered += onset_interval(x).contains(30)
    n = len(seeds)
    assert cusum_hits / n >= 0.9
    assert pelt_hits / n >= 0.9
    assert covered / n >= 0.9
    assert false_alarms / n <= 0.15
