import numpy as np
import pytest

from horizonguard.cli import main
from horizonguard.effects.evaluate import (
    cell_gate,
    evaluate_policies,
    fresh_labels,
    harm_analysis,
    learner_table,
    per_prefix_net,
    recoverability,
    value_with_ci,
)
from horizonguard.eval.leaderboard import load_bench


@pytest.fixture(scope="module")
def bench(small_bench):
    return load_bench(small_bench)


def test_fresh_labels_use_new_seeds_and_cache(bench, small_bench):
    test = bench.part("test")
    fresh = fresh_labels(small_bench, test)
    assert [r["id"] for r in fresh.records] == [r["id"] for r in test.records]
    assert fresh.records[0]["fork_seeds"] != test.records[0]["fork_seeds"]
    assert (small_bench / "fresh-forks.jsonl.gz").exists()
    again = fresh_labels(small_bench, test)
    assert again.records[0]["outcomes"] == fresh.records[0]["outcomes"]


def test_values_and_gate(bench):
    data = bench.part("test")
    zero = np.zeros(len(data), dtype=int)
    assert per_prefix_net(data, zero, 2.0).sum() == 0
    v, lo, hi = value_with_ci(data, np.full(len(data), 4), 1.0, n_boot=100)
    assert lo <= v <= hi
    gate = cell_gate(data, zero, zero, 2.0)
    assert gate["effect_wins"] == 0


def test_full_evaluation_runs(bench, small_bench):
    res = evaluate_policies(bench, small_bench)
    names = [r["policy"] for r in res["policies"]]
    assert names[0] == "effect-aware" and "never intervene" in names
    assert len(learner_table(bench)) == 5
    rec = recoverability(bench)
    assert set(rec) == {"point", "conservative"}
    harm = harm_analysis(bench)
    assert set(harm["compliance_r2"]) == {"reminder", "restatement", "gate", "escalation"}


def test_cli_effects(small_bench, tmp_path):
    out = tmp_path / "effects.md"
    assert main(["effects", "--data", str(small_bench), "--write", str(out)]) == 0
    assert "Policy value on fresh forks" in out.read_text(encoding="utf-8")
