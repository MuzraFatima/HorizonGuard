import pytest

from horizonguard.agents import AGENT_NAMES, CachedBackend, LLMAgent, make_agent
from horizonguard.envs import make_env
from horizonguard.interventions import render
from horizonguard.spec import Intervention


def test_registry_has_at_least_eight_agents_over_five_families():
    assert len(AGENT_NAMES) >= 8
    families = {make_agent(n).family for n in AGENT_NAMES}
    assert len(families - {"scripted"}) >= 5
    assert {make_agent(n).scaffold for n in AGENT_NAMES} >= {"react", "plan", "scripted"}


@pytest.mark.parametrize("name", ["atlas-react", "delta-plan", "scripted"])
def test_agents_act_deterministically_given_seed(name):
    def run():
        env, agent = make_env("portfolio"), make_agent(name)
        env.reset()
        out = []
        for _ in range(20):
            pressure, events = env.pressure(seed=3)
            d = agent.act(env, "obs", pressure, events, seed=3)
            env.step(d.action, seed=3, events=events)
            out.append((d.action.name, d.reasoning))
            if env.done:
                break
        return out

    assert run() == run()


def test_high_pressure_drives_drift_and_compliant_reminder_reduces_it():
    env, agent = make_env("research"), make_agent("delta-react")
    env.reset()
    for _ in range(10):
        agent.act(env, "obs", pressure=1.0, events=[], seed=0)
        env.state.t += 1
    assert agent.state.drift > 0.5
    agent.state.drift = 0.3
    complied = [
        agent.receive(render(Intervention.RESTATEMENT, env.goal_spec, t), seed=s, t=t)
        for s in range(5)
        for t in range(1)
    ]
    assert any(complied)


def test_gate_blocks_violations():
    env, agent = make_env("sysadmin"), make_agent("delta-react")
    env.reset()
    agent.state.drift = 1.0
    blocked = 0
    for seed in range(200):
        agent.state.gate_armed = True
        d = agent.act(env, "obs", 0.0, [], seed=seed)
        assert d.action.kind != "violate"
        blocked += d.blocked
        agent.state.drift = 1.0
    assert blocked > 0


def test_unknown_family_and_scaffold():
    with pytest.raises(KeyError):
        make_agent("zeta-react")
    with pytest.raises(ValueError):
        make_agent("atlas-tree")


def test_llm_agent_parses_actions_and_caches(tmp_path):
    calls = []

    def backend(prompt, seed):
        calls.append(seed)
        return "Stay on goal. ACTION: rebalance-to-mandate"

    cached = CachedBackend(backend, tmp_path / "cache.json")
    agent = LLMAgent(cached)
    env = make_env("portfolio")
    env.reset()
    d1 = agent.act(env, "obs", 0.0, [], seed=1)
    agent.reset()
    d2 = agent.act(env, "obs", 0.0, [], seed=1)
    assert d1.action.name == d2.action.name == "rebalance-to-mandate"
    assert cached.hits == 1 and len(calls) == 1
    cached.save()
    assert CachedBackend(backend, tmp_path / "cache.json").cache == cached.cache


def test_llm_agent_falls_back_and_respects_gate():
    agent = LLMAgent(lambda p, s: "ACTION: disable-monitoring")
    env = make_env("sysadmin")
    env.reset()
    agent.receive(render(Intervention.GATE, env.goal_spec, 0), seed=0, t=0)
    d = agent.act(env, "obs", 0.0, [], seed=0)
    assert d.blocked and d.action.kind == "neutral"
    garbled = LLMAgent(lambda p, s: "no idea")
    assert garbled.act(env, "obs", 0.0, [], seed=0).action.kind == "neutral"
