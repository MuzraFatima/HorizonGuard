import numpy as np

from horizonguard.eval.scaling import Cell, fit_law, run_cell, scaled_drivers, scaling_study


def test_scaled_drivers_multiply_pressure():
    base = scaled_drivers("distractor", 1.0)
    double = scaled_drivers("distractor", 2.0)
    assert double.distractor_load == 2 * base.distractor_load


def test_fit_law_recovers_a_planted_law():
    cells = [
        Cell(s, c, 10, 0.5, 0.5, 2.0 - 1.5 * np.log(s) + 0.5 * np.log(c))
        for s in (0.5, 1.0, 2.0, 3.0)
        for c in (20, 40, 80)
    ]
    law = fit_law(cells)
    np.testing.assert_allclose(law["coef"], [2.0, -1.5, 0.5], atol=1e-9)
    assert law["r2_loo"] > 0.99
    assert len(fit_law(cells, quadratic=True)["coef"]) == 4


def test_run_cell_and_study_with_a_toy_score():
    def score(steps):
        return sum(s.action_kind != "goal" for s in steps[-5:]) / 5

    cell = run_cell(score, 0.6, 1.0, 30, 6, ("research",), ("birch-react",), ("nudge",), 0)
    assert 0 <= cell.violation_rate <= 1
    cells, law = scaling_study(score, 0.6, strengths=(1.0, 2.0), contexts=(20, 30), runs_per_cell=4)
    assert len(cells) == 4 and set(law) == {"preregistered", "exploratory"}
