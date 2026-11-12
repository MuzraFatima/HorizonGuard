import pytest

from horizonguard.prereg import Guarantee, Preregistration, load_preregistration


def test_frozen_values_load():
    prereg = load_preregistration()
    assert prereg.horizon_steps == 20
    assert prereg.disruption_weight == 2.0
    assert prereg.guarantee.alpha == 0.05
    assert prereg.statistics["seeds_per_model"] >= 5


def test_invalid_alpha_is_rejected():
    with pytest.raises(ValueError):
        Preregistration(
            horizon_steps=10,
            disruption_weight=1.0,
            false_alarm_budget_per_1000=5.0,
            recoverability_eps=0.05,
            guarantee=Guarantee(alpha=1.5, delta=0.1, min_rescue_retention=0.5),
            thresholds={},
            statistics={},
        )
