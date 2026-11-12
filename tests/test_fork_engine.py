"""Phase 3 gate: replay determinism and fork consistency on the full stack."""

import numpy as np
import pytest

from horizonguard.agents import LLMAgent, make_agent
from horizonguard.envs import ENVIRONMENTS, make_env
from horizonguard.fork import ForkEngine, fork_seed, prefix_snapshot, run_episode
from horizonguard.spec import Intervention


def actions(traj):
    return [(s.t, s.action, s.reasoning, s.fidelity, s.violated) for s in traj.steps]


@pytest.mark.parametrize("env_name", sorted(ENVIRONMENTS))
def test_same_seed_replays_identically(env_name):
    a = run_episode(make_env(env_name), make_agent("birch-react"), seed=11)
    b = run_episode(make_env(env_name), make_agent("birch-react"), seed=11)
    assert actions(a) == actions(b)


def test_identical_actions_reproduce_identical_transitions():
    env, agent = make_env("codebase"), make_agent("atlas-plan")
    traj = run_episode(env, agent, seed=5)
    replay = make_env("codebase")
    replay.reset()
    by_name = {a.name: a for a in replay.actions}
    for record in traj.steps:
        events = record.extra["events"]
        replay.step(by_name[record.action], seed=5, events=events)
    assert replay.state.goal_progress == env.state.goal_progress
    assert replay.state.violated == env.state.violated


def test_restore_then_continue_matches_uninterrupted_run():
    full = run_episode(make_env("sysadmin"), make_agent("delta-react"), seed=2)
    env, agent = make_env("sysadmin"), make_agent("delta-react")
    head = run_episode(env, agent, seed=2, max_steps=10)
    snap = ForkEngine.snapshot(env, agent, head.run_id, 2)
    run_episode(env, agent, seed=2, max_steps=5, reset=False)  # wander off
    env.restore(snap.env)
    agent.restore(snap.agent)
    tail = run_episode(env, agent, seed=2, run_id=head.run_id, reset=False)
    assert actions(head) + actions(tail) == actions(full)


def test_fork_from_prefix_reproduces_unforked_distribution():
    env, agent = make_env("research"), make_agent("cobalt-react")
    snap, _ = prefix_snapshot(env, agent, seed=4, t=8)
    k, horizon = 64, 20
    result = ForkEngine(horizon).fork(env, agent, snap, arms=(Intervention.NONE,), k=k)
    unforked = []
    for j in range(k):
        e, a = make_env("research"), make_agent("cobalt-react")
        run_episode(e, a, seed=4, max_steps=8)
        cont = run_episode(e, a, seed=fork_seed(4, 8, j), max_steps=horizon, reset=False)
        unforked.append(int(cont.violated))
    np.testing.assert_array_equal(result.outcomes[0], unforked)


def test_forks_are_seed_paired_and_profiled():
    env, agent = make_env("portfolio"), make_agent("birch-react")
    snap, _ = prefix_snapshot(env, agent, seed=0, t=12)
    result = ForkEngine(15).fork(env, agent, snap, k=8)
    assert result.outcomes.shape == (len(Intervention), 8)
    assert len(set(result.seeds)) == 8 and 0 not in result.seeds
    assert 0 < result.steps_run <= len(Intervention) * 8 * 15
    assert 0.0 <= result.meta["early_stop_saving"] < 1.0
    assert np.isfinite(result.compliance(Intervention.RESTATEMENT))


def test_fork_rejects_bad_arms_and_unsnapshottable_agents():
    env, agent = make_env("portfolio"), make_agent("atlas-react")
    snap, _ = prefix_snapshot(env, agent, seed=0, t=3)
    with pytest.raises(ValueError):
        ForkEngine().fork(env, agent, snap, arms=(Intervention.REMINDER,))
    api = LLMAgent(lambda p, s: "ACTION: review-risk-report", counterfactual_ok=False)
    with pytest.raises(ValueError):
        ForkEngine().fork(env, api, snap)


def test_fork_from_violated_prefix_is_always_violated():
    env, agent = make_env("sysadmin"), make_agent("delta-react")
    env.reset()
    agent.reset()
    env.step(env.actions_of("violate")[0], seed=0)
    snap = ForkEngine.snapshot(env, agent, "r", 0)
    assert ForkEngine().fork(env, agent, snap, k=4).outcomes.all()
