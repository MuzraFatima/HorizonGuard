import json

import numpy as np
import pytest

from horizonguard.bench.generate import BenchConfig, generate, load_prefixes, run_specs
from horizonguard.cli import main

TINY = BenchConfig(
    envs=["research", "sysadmin"],
    agents=["birch-react", "ember-plan"],
    drift_types=["nudge", "compaction"],
    k=4,
    horizon=10,
    max_steps=30,
    grid_max_points=3,
    stages=2,
)


@pytest.fixture(scope="module")
def bench(tmp_path_factory):
    out = tmp_path_factory.mktemp("bench")
    manifest = generate(TINY, out)
    return out, manifest, load_prefixes(out)


def test_run_specs_cover_every_cell_with_distinct_seeds():
    specs = run_specs(TINY)
    assert len(specs) == 2 * 2 * 2
    assert len({s.seed for s in specs}) == len(specs)
    assert len({s.run_id for s in specs}) == len(specs)


def test_manifest_and_files(bench):
    out, manifest, prefixes = bench
    assert manifest["runs"] == 8
    assert manifest["prefixes"] == len(prefixes) > 0
    assert (out / "trajectories.jsonl.gz").exists()
    assert json.loads((out / "manifest.json").read_text())["config_hash"] == manifest["config_hash"]


def test_prefix_labels_are_consistent(bench):
    _, _, prefixes = bench
    for p in prefixes:
        y = np.array(p["outcomes"])
        assert y.shape == (5, TINY.k)
        assert p["risk"] == pytest.approx(y[0].mean())
        assert p["risk_lo"] <= p["risk"] <= p["risk_hi"]
        assert p["violation_step"] is None or p["t"] < p["violation_step"]
        for arm in p["arms"]:
            assert arm["benefit"] == pytest.approx(arm["rescue"] - arm["disruption"])


def test_generation_is_reproducible(tmp_path, bench):
    _, _, first = bench
    small = BenchConfig(**{**TINY.__dict__, "stages": 1})
    generate(small, tmp_path)
    again = load_prefixes(tmp_path)
    key = lambda p: p["id"]  # noqa: E731
    assert [p["outcomes"] for p in sorted(again, key=key)] == [
        p["outcomes"] for p in sorted(first, key=key)
    ]


def test_stop_rule_ends_generation_early(tmp_path):
    cfg = BenchConfig(**{**TINY.__dict__, "step_budget": 1, "stages": 4})
    manifest = generate(cfg, tmp_path)
    assert manifest["stopped_early"] == {"stage": 1, "reason": "budget"}
    assert manifest["runs"] < 8


def test_cli_data(tmp_path, capsys):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "seed: 1\nbench:\n  envs: [portfolio]\n  agents: [atlas-react]\n"
        "  drift_types: [nudge]\n  k: 2\n  horizon: 5\n  max_steps: 20\n",
        encoding="utf-8",
    )
    assert main(["data", "--config", str(cfg), "--out", str(tmp_path / "d")]) == 0
    assert "1 runs" in capsys.readouterr().out
