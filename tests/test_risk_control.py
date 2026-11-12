import numpy as np
import pytest

from horizonguard.guarantees.risk_control import (
    crc_threshold,
    hoeffding_bentkus_p,
    ltt_threshold,
    realised,
    run_losses,
)


def synthetic(n_runs, rng, per_run=5):
    """Prefixes whose disruption grows as the policy acts on lower scores."""
    n = n_runs * per_run
    scores = rng.normal(size=n)
    disruption = np.clip(0.02 + 0.25 * (scores < 0) * rng.random(n), 0, 1)
    losses = (rng.random(n) < disruption).astype(float)  # noisy unbiased labels
    rescues = (rng.random(n) < np.clip(0.3 + 0.1 * scores, 0, 1)).astype(float)
    runs = [f"r{i // per_run}" for i in range(n)]
    return scores, losses, rescues, runs


def test_run_losses_average_within_runs():
    scores = np.array([1.0, 0.0, 1.0, 0.0])
    losses = np.array([1.0, 1.0, 0.0, 0.0])
    lr, runs = run_losses(scores, losses, ["a", "a", "b", "b"], np.array([np.inf, 0.5, -1.0]))
    assert runs == ["a", "b"]
    np.testing.assert_allclose(lr, [[0.0, 0.5, 1.0], [0.0, 0.0, 0.0]])


def test_hoeffding_bentkus_p_values():
    assert hoeffding_bentkus_p(0.1, 100, 0.05) == 1.0
    assert hoeffding_bentkus_p(0.0, 500, 0.05) < 1e-6
    assert hoeffding_bentkus_p(0.04, 50, 0.05) > hoeffding_bentkus_p(0.04, 5000, 0.05)


def test_ltt_controls_risk_with_high_probability():
    rng = np.random.default_rng(0)
    alpha, delta, violations, reps = 0.08, 0.1, 0, 150
    for _ in range(reps):
        cal = synthetic(80, rng)
        test = synthetic(400, rng)
        cert = ltt_threshold(*cal[:2], cal[3], alpha, delta)
        held = realised(test[0], test[1], test[2], test[3], cert.lam)
        violations += held["risk"] > alpha
    assert violations / reps <= delta + 0.05


def test_crc_controls_expected_risk():
    rng = np.random.default_rng(1)
    alpha, risks = 0.08, []
    for _ in range(150):
        cal = synthetic(80, rng)
        test = synthetic(400, rng)
        cert = crc_threshold(*cal[:2], cal[3], alpha)
        risks.append(realised(test[0], test[1], test[2], test[3], cert.lam)["risk"])
    assert np.mean(risks) <= alpha + 0.01


def test_tiny_calibration_sets_are_vacuous_not_wrong():
    rng = np.random.default_rng(2)
    cal = synthetic(5, rng)
    cert = crc_threshold(*cal[:2], cal[3], alpha=0.05)
    assert cert.vacuous  # 1 / (n + 1) already exceeds alpha
    assert ltt_threshold(*cal[:2], cal[3], 0.05, 0.1).vacuous


@pytest.mark.parametrize("alpha", [0.05, 0.2])
def test_looser_alpha_never_acts_less(alpha):
    rng = np.random.default_rng(3)
    cal = synthetic(200, rng)
    strict = ltt_threshold(*cal[:2], cal[3], 0.03, 0.1)
    loose = ltt_threshold(*cal[:2], cal[3], alpha, 0.1)
    assert loose.lam <= strict.lam
