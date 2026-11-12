import numpy as np
import pytest

from horizonguard.baselines import (
    CIRRouter,
    EValuatorMonitor,
    FeatureRiskMonitor,
    LatentProbe,
    MarkovMonitor,
    ParadoxRule,
    PeriodicReminder,
    ProbabilityAlarm,
    RuleMonitor,
    always,
    net_benefit,
    never,
    rescue_disruption,
)
from horizonguard.bench.generate import BenchConfig, generate
from horizonguard.eval.dataset import build_prefix_dataset, load_runs
from horizonguard.spec import Intervention

CFG = BenchConfig(
    envs=["research", "sysadmin"],
    agents=["birch-react", "delta-react", "ember-react", "atlas-plan"],
    drift_types=["nudge", "distractor"],
    k=4,
    horizon=10,
    max_steps=30,
    grid_max_points=4,
    stages=1,
)


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    out = tmp_path_factory.mktemp("bl")
    generate(CFG, out)
    return build_prefix_dataset(out), load_runs(out)


@pytest.mark.parametrize(
    "monitor",
    [FeatureRiskMonitor(), FeatureRiskMonitor(groups=("short:",)), RuleMonitor(), LatentProbe()],
    ids=lambda m: m.name,
)
def test_monitors_predict_probabilities(data, monitor):
    ds, runs = data
    monitor.fit(ds, runs)
    p = monitor.predict(ds)
    assert p.shape == (len(ds),) and np.all((p >= 0) & (p <= 1))
    run = next(iter(runs.values()))
    assert 0 <= monitor.score_steps(run.steps[:5]) <= 1


def test_feature_monitor_tracks_rollout_risk(data):
    ds, runs = data
    p = FeatureRiskMonitor().fit(ds, runs).predict(ds)
    assert np.corrcoef(p, ds.risk)[0, 1] > 0.3


def test_markov_monitor_absorption(data):
    ds, runs = data
    m = MarkovMonitor(horizon=10).fit(ds, runs)
    p = m.predict_runs(ds, runs)
    assert np.all((p >= 0) & (p <= 1))
    assert m._absorb["VIOLATED"] == pytest.approx(1.0)


def test_e_valuator_accumulates_evidence(data):
    ds, runs = data
    ev = EValuatorMonitor().fit(ds, runs)
    violating = [r for r in runs.values() if r.violated]
    healthy = [r for r in runs.values() if not r.violated]
    if violating and healthy:
        bad = np.mean([ev.score_steps(r.steps[: r.violation_step]) for r in violating])
        good = np.mean([ev.score_steps(r.steps) for r in healthy])
        assert bad > good


def test_policies_and_net_benefit(data):
    ds, runs = data
    risk = FeatureRiskMonitor().fit(ds, runs).predict(ds)
    assert net_benefit(ds, never(ds), 2.0) == 0.0
    every = always(Intervention.ESCALATION, ds)
    nb = net_benefit(ds, every, 1.0)  # w = 1 makes net benefit equal plain benefit
    assert nb == pytest.approx(ds.arm("benefit")[:, 3].mean())
    r, d, rate = rescue_disruption(ds, every)
    assert 0 <= r <= 1 and 0 <= d <= 1 and rate == 1.0
    alarm = ProbabilityAlarm(0.5).decide(risk, ds)
    assert set(np.unique(alarm)) <= {0, 1}
    periodic = PeriodicReminder(every=10).decide(risk, ds)
    assert all((rec["t"] - 2) % 10 == 0 for rec, a in zip(ds.records, periodic, strict=True) if a)
    rule = ParadoxRule().fit(ds)
    assert all(0 <= rule.cutoff(a) <= 1 for a in rule.rates)
    router = CIRRouter().fit(ds)
    assert router.decide(risk, ds).shape == (len(ds),)
