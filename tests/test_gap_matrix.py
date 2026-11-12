from horizonguard.cli import main
from horizonguard.literature import load_gap_matrix


def test_gap_matrix_passes_phase_one_gate():
    matrix = load_gap_matrix()
    assert matrix.validate() == []


def test_every_pillar_is_covered():
    ids = {p["id"] for p in load_gap_matrix().pillars}
    assert ids == {"P1", "P2", "P3", "P4", "P5", "P6"}


def test_validation_catches_dangling_reference():
    matrix = load_gap_matrix()
    matrix.pillars[0]["closest"] = ["not-a-paper"]
    problems = matrix.validate()
    assert any("not-a-paper" in p for p in problems)


def test_markdown_rendering_lists_every_work():
    matrix = load_gap_matrix()
    table = matrix.to_markdown()
    for wid in matrix.work_ids:
        assert wid in table


def test_unverified_rows_are_reported():
    assert "cir" in load_gap_matrix().unverified()


def test_cli_writes_markdown(tmp_path, capsys):
    out = tmp_path / "gap.md"
    assert main(["gap-matrix", "--write", str(out)]) == 0
    assert "Gap matrix v1" in out.read_text(encoding="utf-8")
    assert "still need a full-text read" in capsys.readouterr().out
