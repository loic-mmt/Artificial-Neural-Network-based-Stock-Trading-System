from pathlib import Path

import pytest

from trading_system.visualization.cli import main


def test_cli_lists_empty_catalogue_without_ui_dependency(tmp_path, capsys):
    assert main(["--artifacts-root", str(tmp_path), "--list"]) == 0
    assert "No saved runs found" in capsys.readouterr().out


def test_cli_missing_optional_dependency_has_install_instruction(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("trading_system.visualization.cli.importlib.util.find_spec", lambda name: None)
    with pytest.raises(SystemExit) as error:
        main(["--artifacts-root", str(tmp_path)])
    assert error.value.code == 2
    assert ".[visualization]" in capsys.readouterr().err


def test_launcher_passes_path_as_argument_and_binds_loopback(tmp_path, monkeypatch):
    monkeypatch.setattr("trading_system.visualization.cli.importlib.util.find_spec", lambda name: object())
    calls = []
    monkeypatch.setattr("trading_system.visualization.cli.subprocess.call", lambda command, env: calls.append((command, env)) or 0)
    assert main(["--artifacts-root", str(tmp_path), "--port", "8512", "--headless"]) == 0
    command, env = calls[0]
    assert command[command.index("--server.address") + 1] == "127.0.0.1"
    assert command[command.index("--server.port") + 1] == "8512"
    assert command[command.index("--theme.base") + 1] == "light"
    assert command[-2:] == ["--artifacts-root", str(tmp_path.resolve())]
    assert Path(env["PYTHONPATH"].split(":")[0]).name == "src"
