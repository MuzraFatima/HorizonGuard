import numpy as np
import pytest

from horizonguard.cli import main
from horizonguard.eval.power import (
    EffectScenario,
    power,
    power_grid,
    smallest_design,
    to_markdown,
)


def test_category_probabilities_are_valid():
    probs = EffectScenario().draw(500, np.random.default_rng(0))
    assert np.all(probs >= -1e-12)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)


def test_power_grows_with_forks_and_prefixes():
    s = EffectScenario()
    small = power(s, k=2, n_prefixes=30, sims=60, seed=1)
    large = power(s, k=12, n_prefixes=200, sims=60, seed=1)
    assert large.heterogeneity_power > small.heterogeneity_power
    assert large.mean_abs_error < small.mean_abs_error


def test_no_heterogeneity_keeps_false_positive_rate_near_alpha():
    # risk still varies across prefixes, so keep rescue well below it
    homogeneous = EffectScenario(
        harm_share=0.0, helpful_rescue=0.1, helpful_disruption=0.05, base_risk=0.5
    )
    result = power(homogeneous, k=8, n_prefixes=100, sims=150, seed=2)
    assert result.heterogeneity_power < 0.12


def test_smallest_design_and_markdown():
    results = power_grid(EffectScenario(), ks=(2, 8), ns=(50, 200), sims=40)
    best = smallest_design(results, target=0.5)
    assert best is not None
    assert "heterogeneity power" in to_markdown(results)
    assert smallest_design(results, target=1.01) is None


def test_invalid_power_arguments():
    with pytest.raises(ValueError):
        power(EffectScenario(), k=0, n_prefixes=10)


def test_cli_power(capsys):
    assert main(["power", "--sims", "20", "--ks", "4", "--ns", "50"]) == 0
    assert "| 4 | 50 |" in capsys.readouterr().out
