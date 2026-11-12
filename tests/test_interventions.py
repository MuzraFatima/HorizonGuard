import pytest

from horizonguard.interventions import GoalSpec, apply_gate, compliance, library, render
from horizonguard.spec import Intervention

SPEC = GoalSpec("Maximise long-run returns", ("Never exceed 5% drawdown", "No leverage"))


def test_messages_are_deterministic_and_graded():
    a = library(SPEC, step=7)
    b = library(SPEC, step=7)
    assert a == b
    assert a[Intervention.NONE].is_noop
    assert SPEC.goal in a[Intervention.REMINDER].text
    assert "No leverage" in a[Intervention.RESTATEMENT].text
    assert "No leverage" not in a[Intervention.REMINDER].text
    assert a[Intervention.GATE].blocks_action and not a[Intervention.GATE].pauses_run
    assert a[Intervention.ESCALATION].pauses_run


def test_restatement_handles_no_constraints():
    msg = render(Intervention.RESTATEMENT, GoalSpec("Do the task"), 0)
    assert "no constraints" in msg.text


def test_gate_blocks_bad_actions():
    assert apply_gate("buy", lambda a: a != "leverage", "hold") == ("buy", False)
    assert apply_gate("leverage", lambda a: a != "leverage", "hold") == ("hold", True)


def test_compliance_is_relative_to_shared_seed_controls():
    treated = [0.8, 0.85, 0.9, 0.9, 0.9, 0.1]
    controls = [[0.6, 0.6, 0.6, 0.6, 0.6], [0.7, 0.7, 0.7, 0.7, 0.7]]
    assert compliance(treated, controls) == pytest.approx(0.87 - 0.65)
    assert compliance([0.5], [0.5]) == 0.0
    with pytest.raises(ValueError):
        compliance([], [0.5])
