import numpy as np
import pytest

from horizonguard.eval.metrics import (
    auroc,
    bootstrap_ci,
    brier,
    ece,
    healthy_mask,
    lead_time_at_budget,
    lead_times,
    rollout_error,
    threshold_at_budget,
)


def test_auroc_matches_definition():
    assert auroc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) == pytest.approx(0.75)
    assert auroc([1, 2, 3], [1, 1, 1]) != auroc([1, 2, 3], [1, 1, 1])  # nan


def test_calibration_metrics():
    p = np.array([0.1, 0.9, 0.5])
    assert brier(p, [0, 1, 0.5]) == pytest.approx(0.02 / 3)
    assert ece([0.2] * 10, [0.2] * 10) == pytest.approx(0.0)
    assert rollout_error([0.2, 0.4], [0.3, 0.2]) == pytest.approx(0.15)


def test_healthy_mask_excludes_pre_violation_window():
    assert healthy_mask(10, 7, 3).tolist() == [True] * 4 + [False] * 6
    assert healthy_mask(4, None, 3).all()


def test_lead_time_at_budget():
    healthy_run = np.zeros(1000)
    rising = np.concatenate([np.zeros(20), np.linspace(0.1, 1.0, 10)])
    scores = [healthy_run, rising]
    viol = [None, 30]
    thr = threshold_at_budget(scores, viol, budget_per_1000=0.0, horizon=10)
    assert thr > 0
    assert lead_times(scores, viol, thr).tolist() == [10.0]
    out = lead_time_at_budget(scores, viol, budget_per_1000=0.0, horizon=10)
    assert out["median_lead"] == 10 and out["detection_rate"] == 1.0
    # with a short horizon the early rise counts as healthy, so alarms come later
    assert lead_time_at_budget(scores, viol, 0.0, horizon=5)["median_lead"] == 5


def test_budget_allows_some_false_alarms():
    rng = np.random.default_rng(0)
    healthy = [rng.random(500) for _ in range(4)]
    thr = threshold_at_budget(healthy, [None] * 4, budget_per_1000=10, horizon=5)
    alarms = sum((h >= thr).sum() for h in healthy)
    assert alarms <= 20


def test_bootstrap_ci_brackets_point():
    rng = np.random.default_rng(1)
    groups = [rng.normal(1.0, 1.0, 5) for _ in range(50)]
    point, lo, hi = bootstrap_ci(groups, lambda g: float(np.mean(np.concatenate(g))), n=300)
    assert lo < point < hi
    assert 0.1 < hi - lo < 1.0  # roughly 4 standard errors of a 250-draw mean
