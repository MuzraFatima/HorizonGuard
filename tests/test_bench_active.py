import numpy as np
import pytest

from horizonguard.bench.active import allocate, control_variate, priorities, shrink
from horizonguard.bench.generate import BenchConfig, generate, load_prefixes

TINY = BenchConfig(
    envs=["research"],
    agents=["birch-react", "cobalt-react"],
    drift_types=["nudge"],
    k=4,
    horizon=10,
    max_steps=30,
    grid_max_points=3,
    stages=1,
)


def test_shrinkage_pulls_noisy_estimates_toward_the_mean():
    rng = np.random.default_rng(0)
    truth = rng.normal(0.1, 0.05, 400)
    se = np.full(400, 0.2)
    noisy = truth + rng.normal(0, se)
    out = shrink(noisy, se)
    assert np.mean((out.posterior_mean - truth) ** 2) < np.mean((noisy - truth) ** 2)
    assert out.prior_mean == pytest.approx(0.1, abs=0.03)
    assert np.all(out.posterior_sd <= se)


def test_control_variate_reduces_variance_when_correlated():
    rng = np.random.default_rng(1)
    c = rng.normal(0, 1, 500)
    y = 2.0 + 0.9 * c + rng.normal(0, 0.3, 500)
    plain_se = y.std(ddof=1) / np.sqrt(y.size)
    mean, se = control_variate(y, c, mu_c=0.0)
    assert mean == pytest.approx(2.0, abs=0.05)
    assert se < plain_se / 2


def test_priorities_favour_uncertain_prefixes_near_threshold():
    sure = {"k": 6, "arms": [{"arm": 1, "benefit": 0.6, "benefit_se": 0.05}]}
    unsure = {"k": 6, "arms": [{"arm": 1, "benefit": 0.02, "benefit_se": 0.2}]}
    p = priorities([sure, unsure])
    assert p[1] > p[0]


def test_top_up_adds_fresh_seeds_and_keeps_labels_consistent(tmp_path):
    generate(TINY, tmp_path)
    records = load_prefixes(tmp_path)
    updated, chosen = allocate(records, TINY, budget_prefixes=2, extra_k=4)
    assert len(chosen) == 2
    for i in chosen:
        new, old = updated[i], records[i]
        assert new["k"] == 8
        assert new["outcomes"][0][:4] == old["outcomes"][0]
        assert len(set(new["fork_seeds"])) == 8
        y = np.array(new["outcomes"])
        assert new["risk"] == pytest.approx(y[0].mean())
