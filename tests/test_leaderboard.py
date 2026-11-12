import math

import pytest

from horizonguard.baselines import FeatureRiskMonitor, MarkovMonitor
from horizonguard.cli import main
from horizonguard.eval.leaderboard import (
    leaderboard,
    load_bench,
    paradox_check,
    shortcut_check,
    to_markdown,
)


@pytest.fixture(scope="module")
def bench_dir(small_bench):
    return small_bench


def test_leaderboard_rows(bench_dir):
    bench = load_bench(bench_dir)
    rows = leaderboard(bench, [FeatureRiskMonitor(), MarkovMonitor(horizon=10)])
    names = [r["monitor"] for r in rows]
    assert names == ["probability", "markov", "risk-controlled"]
    assert 0 <= rows[0]["test:rollout_mae"] <= 1
    assert "ood_test:auroc" in rows[0]
    assert "| probability |" in to_markdown(rows)


def test_reality_checks_run(bench_dir):
    bench = load_bench(bench_dir)
    sc = shortcut_check(bench, n_boot=100)
    assert sc["mae_gain_ci"][0] <= sc["mae_gain"] <= sc["mae_gain_ci"][1]
    px = paradox_check(bench)
    assert {row["agent"] for row in px["agents"]} >= {"birch-react", "ember-react"}
    assert not math.isnan(px["overall_net_benefit"])


def test_cli_baselines(bench_dir, tmp_path, capsys):
    out = tmp_path / "lb.md"
    assert main(["baselines", "--data", str(bench_dir), "--write", str(out), "--quick"]) == 0
    text = out.read_text(encoding="utf-8")
    assert "Reality check 1" in text and "Reality check 2" in text
