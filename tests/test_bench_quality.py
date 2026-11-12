import pytest

from horizonguard.bench.generate import BenchConfig, generate, load_prefixes
from horizonguard.bench.quality import (
    arm_noise,
    gate_passes,
    replicate_agreement,
    report,
    within_variance,
)
from horizonguard.cli import main

CFG = BenchConfig(
    envs=["research", "sysadmin"],
    agents=["birch-react", "delta-react", "ember-react"],
    drift_types=["nudge", "distractor"],
    k=6,
    horizon=12,
    max_steps=36,
    grid_max_points=4,
    stages=1,
)


@pytest.fixture(scope="module")
def records(tmp_path_factory):
    out = tmp_path_factory.mktemp("q")
    generate(CFG, out)
    return out, load_prefixes(out)


def test_within_variance_is_zero_only_without_discordant_pairs():
    rec = {"k": 6, "arms": [{"rescue": 0.0, "disruption": 0.0}, {"rescue": 0.5, "disruption": 0.0}]}
    assert within_variance(rec, 0) == 0.0
    assert within_variance(rec, 1) == pytest.approx((0.5 - 0.25) / 5)


def test_noise_report_detects_heterogeneity(records):
    _, recs = records
    noise = arm_noise(recs)
    assert [a.arm for a in noise] == ["reminder", "restatement", "gate", "escalation"]
    assert gate_passes(noise)
    assert noise[-1].signal_sd > 0


def test_replicates_agree_and_noise_is_predicted(records):
    _, recs = records
    agreement = replicate_agreement(recs, CFG, n=20, arm=3)
    assert agreement.risk_corr > 0.5
    assert agreement.benefit_noise_sd < 2 * agreement.predicted_benefit_se
    assert "PASS" in report(recs, agreement)


def test_cli_audit(records, capsys):
    out, _ = records
    assert main(["audit", "--data", str(out), "--replicates", "0"]) == 0
    assert "label-noise report" in capsys.readouterr().out
