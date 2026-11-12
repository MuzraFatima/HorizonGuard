import pytest

import horizonguard
from horizonguard.cli import main


def test_version_is_exposed():
    assert horizonguard.__version__ == "1.0.0"


def test_cli_without_command_prints_help(capsys):
    assert main([]) == 0
    assert "goal drift" in capsys.readouterr().out


def test_cli_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "1.0.0" in capsys.readouterr().out
