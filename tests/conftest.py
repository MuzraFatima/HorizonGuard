"""Shared fixtures: one small generated benchmark reused across modules."""

import pytest

from horizonguard.bench.generate import BenchConfig, generate

SMALL = BenchConfig(
    envs=["research", "sysadmin"],
    agents=["birch-react", "delta-react", "cobalt-react", "ember-react"],
    drift_types=["nudge", "distractor", "context"],
    replicates=2,
    k=4,
    horizon=10,
    max_steps=30,
    grid_max_points=4,
    stages=1,
)


@pytest.fixture(scope="session")
def small_bench(tmp_path_factory):
    out = tmp_path_factory.mktemp("small-bench")
    generate(SMALL, out)
    return out
