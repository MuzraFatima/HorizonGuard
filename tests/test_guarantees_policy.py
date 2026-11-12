import numpy as np
import pytest

from horizonguard.cli import main
from horizonguard.eval.leaderboard import load_bench
from horizonguard.guarantees.evaluate import coverage_study, held_out, time_uniform
from horizonguard.guarantees.policy import CertifiedPolicy


@pytest.fixture(scope="module")
def bench(small_bench):
    return load_bench(small_bench)


def test_certified_policy_requires_certificate(bench):
    pol = CertifiedPolicy(alpha=0.2).fit(bench.part("train"))
    with pytest.raises(RuntimeError):
        pol.decide(bench.part("test"))
    cert = pol.certify(bench.part("calibration"))
    arms = pol.decide(bench.part("test"))
    assert set(np.unique(arms)) <= {0, 1, 2, 3, 4}
    if cert.vacuous:
        assert (arms == 0).all()


def test_crc_and_ltt_certificates(bench):
    for method in ("crc", "ltt"):
        pol = CertifiedPolicy(method=method, alpha=0.3, delta=0.2).fit(bench.part("train"))
        cert = pol.certify(bench.part("calibration"))
        assert cert.method == method and cert.n_runs > 0
        assert cert.vacuous or cert.empirical_risk <= 0.3


def test_study_functions_run(bench, small_bench):
    rows = held_out(bench, small_bench, alphas=(0.2,))
    assert [r["method"] for r in rows] == ["crc", "ltt", "uncertified"]
    cov = coverage_study(bench, alpha=0.2, reps=5)
    assert 0 <= cov["ltt_violation_rate"] <= 1
    tu = time_uniform(bench, stride=3)
    assert 0 <= tu["eprocess_run_fpr"] <= 1


def test_cli_guarantees(small_bench, tmp_path):
    out = tmp_path / "g.md"
    assert main(["guarantees", "--data", str(small_bench), "--write", str(out), "--quick"]) == 0
    assert "Held-out disruption" in out.read_text(encoding="utf-8")
