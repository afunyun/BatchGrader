import sys

import pytest

import batchgrader.cli as cli


def test_main_help(monkeypatch, capsys):
    monkeypatch.setattr(cli, "setup_logging", lambda log_dir: None)
    monkeypatch.setattr(cli, "load_config", lambda config_file: {})
    sys_args = ["prog", "--help"]
    monkeypatch.setattr(sys, "argv", sys_args)
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "BatchGrader CLI" in captured.out


@pytest.mark.parametrize(
    "mode, func_name",
    [
        ("batch", "run_batch_processing"),
        ("count", "run_count_mode"),
        ("split", "run_split_mode"),
    ],
)
def test_main_modes(monkeypatch, mode, func_name):
    monkeypatch.setattr(cli, "setup_logging", lambda log_dir: None)
    monkeypatch.setattr(cli, "load_config", lambda config_file: {})
    calls = {}

    def record_call(args, config):
        calls[func_name] = True

    monkeypatch.setattr(cli, func_name, record_call)
    sys_args = ["prog", "--input-file", "input.txt", "--mode", mode]
    monkeypatch.setattr(sys, "argv", sys_args)
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0
    assert func_name in calls


def test_main_unsupported_mode(monkeypatch):
    monkeypatch.setattr(cli, "setup_logging", lambda log_dir: None)
    monkeypatch.setattr(cli, "load_config", lambda config_file: {})
    sys_args = ["prog", "--input-file", "input.txt", "--mode", "foo"]
    monkeypatch.setattr(sys, "argv", sys_args)
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
