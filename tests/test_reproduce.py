from horizonguard.cli import REPORTS, main


def test_reproduce_writes_every_report_and_keeps_readings(small_bench, tmp_path):
    (tmp_path / "effects.md").write_text(
        "# old\n\n## Verdict and reading\n\nHand-written notes.\n", encoding="utf-8"
    )
    assert main(["reproduce", "--data", str(small_bench), "--out", str(tmp_path), "--quick"]) == 0
    for _, target in REPORTS:
        assert (tmp_path / target.split("/")[-1]).exists()
    effects = (tmp_path / "effects.md").read_text(encoding="utf-8")
    assert "Policy value on fresh forks" in effects and "Hand-written notes." in effects
    assert (small_bench / "monitor.pkl").exists()
