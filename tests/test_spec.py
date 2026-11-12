from horizonguard.spec import Intervention, State, Transition, states_from_labels


def test_violated_is_the_only_absorbing_state():
    assert [s.absorbing for s in State] == [False, False, True]


def test_transitions_map_between_states():
    assert Transition.between(State.ALIGNED, State.DRIFTING) is Transition.ONSET
    assert Transition.between(State.DRIFTING, State.ALIGNED) is Transition.RECOVERY
    assert Transition.between(State.VIOLATED, State.ALIGNED) is None
    assert Transition.ABRUPT.source is State.ALIGNED


def test_interventions_are_graded():
    assert Intervention.NONE < Intervention.REMINDER < Intervention.ESCALATION
    assert Intervention.NONE not in Intervention.active()


def test_states_from_labels():
    states = states_from_labels(onset=2, violation=5, length=7)
    assert states == [0, 0, 1, 1, 1, 2, 2]
    assert states_from_labels(None, None, 3) == [State.ALIGNED] * 3
    assert states_from_labels(None, 1, 3) == [0, 2, 2]
