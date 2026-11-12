import pytest

from horizonguard.envs.base import Action, DriftDrivers
from horizonguard.envs.suite import ENVIRONMENTS, make_env
from horizonguard.labels import FidelityScorer


@pytest.mark.parametrize("name", sorted(ENVIRONMENTS))
def test_every_env_has_all_action_kinds(name):
    env = make_env(name)
    for kind in ("goal", "side", "violate", "neutral"):
        assert env.actions_of(kind), f"{name} lacks {kind} actions"
    assert len(env.constraint_texts) >= 3
    assert env.goal_spec.goal == env.goal_text


@pytest.mark.parametrize("name", sorted(ENVIRONMENTS))
def test_goal_actions_finish_the_task(name):
    env = make_env(name, max_steps=200)
    env.reset()
    goal = env.actions_of("goal")[0]
    steps = 0
    while not env.done:
        env.step(goal, seed=0)
        steps += 1
    assert env.succeeded and steps < 200


def test_violation_ends_episode_and_zeros_fidelity():
    env = make_env("sysadmin")
    env.reset()
    env.step(env.actions_of("goal")[0], seed=0)
    env.step(env.actions_of("violate")[0], seed=0)
    assert env.done and not env.succeeded
    scorer = FidelityScorer(env.constraints())
    assert scorer.score(env.state.history) == 0.0
    assert not env.is_allowed(env.actions_of("violate")[0])


def test_side_work_lowers_fidelity_without_violation():
    env = make_env("research")
    env.reset()
    for _ in range(4):
        env.step(env.actions_of("side")[0], seed=0)
    scorer = FidelityScorer(env.constraints())
    assert 0 < scorer.score(env.state.history) < 1


def test_snapshot_restore_and_pressure_are_seeded():
    env = make_env("portfolio", drivers=DriftDrivers(nudge_rate=0.5, goal_switch_step=3))
    env.reset()
    for _ in range(3):
        env.step(env.actions_of("goal")[0], seed=7)
    snap = env.snapshot()
    p1, e1 = env.pressure(seed=7)
    assert "goal-switch" in e1
    env.step(env.actions_of("side")[0], seed=7)
    env.restore(snap)
    assert env.pressure(seed=7) == (p1, e1)
    assert env.state.t == 3


def test_unknown_action_and_env_rejected():
    env = make_env("codebase")
    with pytest.raises(ValueError):
        env.step(Action("rm-rf", "violate"), seed=0)
    with pytest.raises(KeyError):
        make_env("chess")
    with pytest.raises(ValueError):
        Action("x", "weird")
