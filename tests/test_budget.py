import pytest

from horizonguard.budget import BudgetPlan, CostModel, blueprint_example
from horizonguard.cli import main


def test_blueprint_example_matches_section_seven():
    plan = blueprint_example()
    assert plan.continuations == 90_000
    assert plan.agent_steps == 1_800_000


def test_early_stopping_reduces_steps_not_continuations():
    plan = BudgetPlan(10, 5, 2, 6, 20, early_stop_fraction=0.25)
    assert plan.continuations == 900
    assert plan.agent_steps == 13_500


def test_stages_cover_every_run():
    stages = blueprint_example().staged(stages=3)
    assert sum(s["runs"] for s in stages) == 1000
    assert stages[-1]["cumulative_runs"] == 1000


def test_cost_model_mixes_shares():
    cost = CostModel(open_weights_per_step=1.0, api_per_step=3.0, open_weights_share=0.5)
    assert cost.per_step() == pytest.approx(2.0)


@pytest.mark.parametrize(
    "bad", [{"forked_runs": 0}, {"strengths": -1}, {"early_stop_fraction": 1.0}]
)
def test_invalid_plans_rejected(bad):
    args = {
        "forked_runs": 1,
        "grid_prefixes": 1,
        "strengths": 1,
        "continuations_per_arm": 1,
        "horizon": 1,
    }
    args.update(bad)
    with pytest.raises(ValueError):
        BudgetPlan(**args)


def test_cli_budget(capsys):
    assert main(["budget"]) == 0
    out = capsys.readouterr().out
    assert "90000" in out.replace(",", "")
