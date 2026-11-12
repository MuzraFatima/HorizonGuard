from horizonguard.cli import main
from horizonguard.eval.pilot import run_pilot, to_markdown
from horizonguard.logging import group_runs, read_jsonl


def test_pilot_base_rates_are_usable_everywhere():
    env_rates, agent_rates, profile = run_pilot(runs_per_env=24)
    assert {r.group for r in env_rates} == {"portfolio", "codebase", "research", "sysadmin"}
    assert all(r.usable for r in env_rates), to_markdown(env_rates, "env")
    scripted = next(r for r in agent_rates if r.group == "scripted")
    assert scripted.violation_rate == 0.0 and scripted.drift_rate == 0.0
    # families must differ, otherwise transfer questions are empty
    rates = [r.violation_rate for r in agent_rates if r.group != "scripted"]
    assert max(rates) - min(rates) > 0.3
    assert profile["steps"] > 0


def test_pilot_writes_trajectories(tmp_path):
    out = tmp_path / "pilot.jsonl"
    run_pilot(runs_per_env=2, envs=("portfolio",), out=out)
    runs = group_runs(read_jsonl(out))
    assert len(runs) == 2


def test_cli_pilot(capsys):
    assert main(["pilot", "--runs", "12"]) == 0
    assert "| portfolio |" in capsys.readouterr().out
