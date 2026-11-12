import numpy as np
import pytest

from horizonguard.bench.generate import BenchConfig, generate
from horizonguard.eval.dataset import build_prefix_dataset, load_runs, step_scores

TINY = BenchConfig(
    envs=["research"],
    agents=["birch-react", "ember-react"],
    drift_types=["nudge", "distractor"],
    k=4,
    horizon=10,
    max_steps=30,
    grid_max_points=3,
    stages=1,
)


@pytest.fixture(scope="module")
def bench(tmp_path_factory):
    out = tmp_path_factory.mktemp("ds")
    generate(TINY, out)
    return out


def test_dataset_rows_match_prefixes_and_cache(bench):
    ds = build_prefix_dataset(bench)
    assert ds.x.shape == (len(ds), len(ds.feature_names))
    assert ds.arm("benefit").shape == (len(ds), 4)
    assert list(bench.glob("features-*.npz"))
    again = build_prefix_dataset(bench)
    np.testing.assert_array_equal(again.x, ds.x)


def test_subsets_and_shortcut_dropping(bench):
    ds = build_prefix_dataset(bench)
    runs = sorted(set(ds.meta["run_id"]))[:2]
    sub = ds.select_runs(runs)
    assert set(sub.meta["run_id"]) == set(runs)
    dropped = ds.drop_groups(["short:"])
    assert dropped.shape[1] == ds.x.shape[1] - 3


def test_step_scores_cover_every_step(bench):
    runs = list(load_runs(bench).values())
    scores = step_scores(runs, lambda steps: float(len(steps)), stride=2)
    assert [len(s) for s in scores] == [len(r) for r in runs]
    assert scores[0][1] == scores[0][0] == 1.0
