import pytest

from horizonguard.monitor.closed_loop import (
    SupportDeskEnv,
    ab_test,
    demo_agent,
    demo_env,
    fresh_agent,
    fresh_env,
    run_closed_loop,
)
from horizonguard.monitor.live import LiveMonitor, load_or_build


@pytest.fixture(scope="module")
def monitor(small_bench, tmp_path_factory):
    cache = tmp_path_factory.mktemp("art") / "monitor.pkl"
    artifacts = load_or_build(small_bench, cache)
    assert cache.exists()
    assert load_or_build(small_bench, cache).certificate == artifacts.certificate
    return LiveMonitor(artifacts)


def test_fresh_env_and_agent_are_not_in_training():
    from horizonguard.agents import FAMILIES
    from horizonguard.envs import ENVIRONMENTS

    assert SupportDeskEnv.name not in ENVIRONMENTS
    assert "fjord" not in FAMILIES
    assert fresh_agent(1).scaffold == "plan"
    assert fresh_env(0).name == "support-desk"


def test_assessments_have_dashboard_fields(monitor):
    ep = run_closed_loop(fresh_env(3, 25), fresh_agent(3), 3, "horizonguard", monitor)
    first = ep["timeline"][0]
    for key in (
        "risk",
        "p_drifting",
        "net_benefit",
        "best_arm",
        "certified",
        "alert",
        "fix_by",
        "evidence",
        "fidelity",
    ):
        assert key in first
    assert 0 <= first["risk"] <= 1
    assert set(first["net_benefit"]) == {"reminder", "restatement", "gate", "escalation"}


def test_no_monitor_arm_never_intervenes(monitor):
    ep = run_closed_loop(fresh_env(4, 25), fresh_agent(4), 4, "none", monitor)
    assert ep["interventions"] == 0


def test_ab_test_shares_seeds(monitor):
    res = ab_test(monitor, seeds=range(3), max_steps=20)
    assert set(res.summary) == {"none", "probability", "horizonguard"}
    assert res.summary["none"]["rescued"] == 0


def test_demo_factories():
    assert demo_agent("fjord-plan").family == "fjord"
    assert demo_agent("birch-react").family == "birch"
    assert demo_env("support-desk", 0).name == "support-desk"
    assert demo_env("research", 0).name == "research"
