import numpy as np
import pytest

from horizonguard.bench.splits import leave_one_out
from horizonguard.eval.dataset import build_prefix_dataset
from horizonguard.transfer.coldstart import cold_start_curve, weighted_crc
from horizonguard.transfer.matrix import evaluate_split, summarise, transfer_matrix


@pytest.fixture(scope="module")
def data(small_bench):
    return build_prefix_dataset(small_bench)


def test_evaluate_split_reports_all_three_components(data):
    split = next(s for s in leave_one_out(data.records, "family") if s.ood_test)
    row = evaluate_split(data, split)
    for key in ("risk_auroc_drop", "effect_mae_ratio", "guarantee_risk_ood", "guarantee_holds_ood"):
        assert key in row


def test_transfer_matrix_and_summary(data):
    rows = transfer_matrix(data, axes=("env", "drift_type"))
    assert {r["axis"] for r in rows} == {"env", "drift_type"}
    s = summarise(rows, max_drop=0.1)
    assert s["splits"] == len(rows)
    assert 0 <= s["share_guarantee_holds"] <= 1


def test_weighted_crc_is_more_conservative_with_skewed_weights():
    rng = np.random.default_rng(0)
    scores = rng.normal(size=400)
    losses = (rng.random(400) < 0.05).astype(float)
    runs = [f"r{i // 4}" for i in range(400)]
    even = {f"r{j}": 1.0 for j in range(100)}
    skew = {f"r{j}": (10.0 if j == 0 else 1.0) for j in range(100)}
    assert weighted_crc(scores, losses, runs, skew, 0.2) >= weighted_crc(
        scores, losses, runs, even, 0.2
    )


def test_cold_start_curve(data):
    rows = cold_start_curve(data, "family", "birch", sizes=(0, 2), alpha=0.2, reps=3)
    assert rows and rows[0]["n_target"] == 0
    assert {"pooled_violation", "weighted_vacuous", "effect_mae"} <= set(rows[0])


def test_cli_transfer(small_bench, tmp_path):
    from horizonguard.cli import main

    out = tmp_path / "t.md"
    assert main(["transfer", "--data", str(small_bench), "--write", str(out), "--quick"]) == 0
    assert "Transfer matrix" in out.read_text(encoding="utf-8")
