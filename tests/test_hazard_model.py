import numpy as np
import pytest

from horizonguard.cli import main
from horizonguard.eval.leaderboard import load_bench
from horizonguard.hazard.comparators import LogisticHazard, PCAAnomaly, SingleEventClassifier
from horizonguard.hazard.evaluate import (
    attribution,
    curve_markdown,
    lead_curve,
    lead_gain,
    score_monitor,
)
from horizonguard.hazard.model import MultiStateHazard


@pytest.fixture(scope="module")
def bench_dir(small_bench):
    return small_bench


@pytest.fixture(scope="module")
def fitted(bench_dir):
    bench = load_bench(bench_dir)
    return bench, MultiStateHazard(horizon=10).fit(bench.part("train"), bench.runs)


def test_transition_rows_are_distributions(fitted):
    bench, model = fitted
    out = model.hazards(bench.data.x[:20])
    np.testing.assert_allclose(out.from_aligned.sum(axis=1), 1.0, atol=1e-9)
    np.testing.assert_allclose(out.from_drifting.sum(axis=1), 1.0, atol=1e-9)
    p = out.matrices()
    np.testing.assert_allclose(p.sum(axis=2), 1.0, atol=1e-9)
    assert np.all(p[:, 2, 2] == 1.0)


def test_risk_is_monotone_in_horizon_and_bounded(fitted):
    bench, model = fitted
    multi = model.multi_horizon(bench.data.x[:30], horizons=(1, 5, 10, 20))
    assert np.all(np.diff(multi, axis=1) >= -1e-12)
    r = model.predict(bench.data)
    assert np.all((r >= 0) & (r <= 1))
    assert np.all(model.onset_risk(bench.data.x[:5]) >= 0)


def test_hazard_tracks_rollout_risk(fitted):
    bench, model = fitted
    test = bench.part("test")
    assert np.corrcoef(model.predict(test), test.risk)[0, 1] > 0.3


@pytest.mark.parametrize("cls", [SingleEventClassifier, LogisticHazard, PCAAnomaly])
def test_comparators_fit_and_score(fitted, cls):
    bench, _ = fitted
    m = cls().fit(bench.part("train"), bench.runs)
    p = m.predict(bench.part("test"))
    assert np.all((p >= 0) & (p <= 1))


def test_curves_gain_and_attribution(fitted):
    bench, model = fitted
    a = score_monitor(MultiStateHazard(horizon=10), bench, stride=3)
    b = score_monitor(SingleEventClassifier(horizon=10), bench, stride=3)
    rows = lead_curve([a, b], budgets=(5.0, 20.0))
    assert "lead@5" in rows[0]
    point, lo, hi = lead_gain(a, b, 20.0, n_boot=200)
    assert lo <= point <= hi
    assert "| multi-state-hazard |" in curve_markdown(rows, budgets=(5.0, 20.0))
    attr = attribution(model, bench.data.feature_names)
    assert set(attr) == {"onset", "violation"} and attr["onset"]["top"]


def test_cli_hazard(bench_dir, tmp_path):
    out = tmp_path / "hz.md"
    assert main(["hazard", "--data", str(bench_dir), "--write", str(out), "--quick"]) == 0
    assert "Lead-time curve" in out.read_text(encoding="utf-8")
