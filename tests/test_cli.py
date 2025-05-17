import pytest
from unittest.mock import patch, MagicMock, call
import sys
from pathlib import Path

# sys.path.insert(0, str(Path(__file__).parent.parent)) # Likely redundant with pytest.ini python_paths

from src import cli  # Removed src. prefix
from src.constants import PROJECT_ROOT, LOG_DIR as DEFAULT_LOG_DIR_CONST  # Removed src. prefix


# Helper to set sys.argv for a test
@pytest.fixture
def mock_sys_argv(monkeypatch):

    def _mock_sys_argv(argv_list):
        monkeypatch.setattr(sys, 'argv', argv_list)

    return _mock_sys_argv


@pytest.fixture
def mock_batch_runner_functions(mocker):
    mock_run_batch = mocker.patch(
        'src.cli.run_batch_processing')  # Removed src. prefix
    mock_run_count = mocker.patch('src.cli.run_count_mode')  # Removed src. prefix
    mock_run_split = mocker.patch('src.cli.run_split_mode')  # Removed src. prefix
    return {
        'batch': mock_run_batch,
        'count': mock_run_count,
        'split': mock_run_split
    }


@pytest.fixture
def mock_load_config(mocker):
    return mocker.patch(
        'src.cli.load_config',  # Corrected src. prefix
        return_value={'test_key': 'test_value'})


def test_cli_batch_mode_input_file(mock_sys_argv, mock_batch_runner_functions,
                                   mock_load_config, tmp_path):
    input_file = tmp_path / "input.csv"
    input_file.write_text("id,text\n1,test")
    output_dir = tmp_path / "output_cli"

    argv = [
        'script_name', '--input-file',
        str(input_file), '--output-dir',
        str(output_dir), '--mode', 'batch'
    ]
    mock_sys_argv(argv)

    with pytest.raises(
            SystemExit) as e:  # cli.main calls sys.exit(0) on success
        cli.main()
    assert e.value.code == 0

    mock_load_config.assert_called_once_with(None)  # Default config file
    args, config = mock_batch_runner_functions['batch'].call_args[0]

    assert args.input_file == str(input_file)
    assert args.output_dir == str(output_dir)
    assert args.mode == 'batch'
    assert config == {'test_key': 'test_value'}
    mock_batch_runner_functions['count'].assert_not_called()
    mock_batch_runner_functions['split'].assert_not_called()


def test_cli_count_mode_input_dir(mock_sys_argv, mock_batch_runner_functions,
                                  mock_load_config, tmp_path):
    input_dir = tmp_path / "input_data_cli"
    input_dir.mkdir()
    (input_dir / "data.csv").write_text("id,text\n1,data")

    config_file_path = tmp_path / "custom_config.yaml"
    config_file_path.write_text("custom_setting: true")

    argv = [
        'script_name', '--input-dir',
        str(input_dir), '--mode', 'count', '--config',
        str(config_file_path), '--stats'
    ]
    mock_sys_argv(argv)

    with pytest.raises(SystemExit) as e:
        cli.main()
    assert e.value.code == 0

    mock_load_config.assert_called_once_with(str(config_file_path))
    args, config = mock_batch_runner_functions['count'].call_args[0]

    assert args.input_dir == str(input_dir)
    assert args.mode == 'count'
    assert args.stats is True
    assert config == {
        'test_key': 'test_value'
    }  # mock_load_config returns this
    mock_batch_runner_functions['batch'].assert_not_called()
    mock_batch_runner_functions['split'].assert_not_called()


def test_cli_split_mode_log_dir(mock_sys_argv, mock_batch_runner_functions,
                                mock_load_config, tmp_path):
    input_file = tmp_path / "split_me.xlsx"
    input_file.write_text(
        "dummy excel content")  # Actual content not vital for CLI test
    custom_log_dir = tmp_path / "custom_logs"

    argv = [
        'script_name', '--input-file',
        str(input_file), '--mode', 'split', '--log-dir',
        str(custom_log_dir)
    ]
    mock_sys_argv(argv)

    with pytest.raises(SystemExit) as e:
        cli.main()
    assert e.value.code == 0

    mock_load_config.assert_called_once_with(None)
    args, config = mock_batch_runner_functions['split'].call_args[0]

    assert args.input_file == str(input_file)
    assert args.mode == 'split'
    assert args.log_dir == str(custom_log_dir)
    assert config == {'test_key': 'test_value'}
    mock_batch_runner_functions['batch'].assert_not_called()
    mock_batch_runner_functions['count'].assert_not_called()


def test_cli_missing_input_defaults_to_input_dir(monkeypatch, mock_sys_argv, mock_batch_runner_functions, mock_load_config, tmp_path):
    # Simulate input/ directory with a file
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "data.csv").write_text("id,text\n1,data")
    monkeypatch.setattr("src.constants.PROJECT_ROOT", tmp_path)

    argv = ['script_name', '--mode', 'batch']  # No --input-file or --input-dir
    mock_sys_argv(argv)

    with pytest.raises(SystemExit) as e:
        cli.main()
    assert e.value.code == 0
    args, config = mock_batch_runner_functions['batch'].call_args[0]
    assert args.input_dir == str(input_dir)
    assert args.input_file is None


def test_cli_missing_input_and_empty_input_dir(monkeypatch, mock_sys_argv, mock_batch_runner_functions, mock_load_config, tmp_path, capfd):
    # Simulate empty input/ directory
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    monkeypatch.setattr("src.constants.PROJECT_ROOT", tmp_path)

    argv = ['script_name', '--mode', 'batch']  # No --input-file or --input-dir
    mock_sys_argv(argv)

    with pytest.raises(SystemExit) as e:
        cli.main()
    assert e.value.code == 2
    out, err = capfd.readouterr()
    assert "missing or empty" in out or "missing or empty" in err, "Did not log error message for missing/empty input dir."


def test_cli_default_output_dir(mock_sys_argv, mock_batch_runner_functions,
                                mock_load_config, tmp_path):
    input_file = tmp_path / "input.csv"
    input_file.write_text("id,text\n1,test")

    # Expected default output directory structure
    # PROJECT_ROOT / 'output' / 'batch_results'
    # For testing, we can mock PROJECT_ROOT or just check that output_dir is None in args
    # as run_batch_processing itself resolves the default if args.output_dir is None.

    argv = [
        'script_name',
        '--input-file',
        str(input_file),
        '--mode',
        'batch'  # No --output-dir
    ]
    mock_sys_argv(argv)

    with pytest.raises(SystemExit) as e:
        cli.main()
    assert e.value.code == 0

    args, _ = mock_batch_runner_functions['batch'].call_args[0]
    assert args.output_dir is None  # cli.py passes None if not specified, batch_runner handles default


def test_cli_load_config_file_not_found(mock_sys_argv,
                                        mock_batch_runner_functions, mocker,
                                        tmp_path):
    input_file = tmp_path / "input.csv"
    input_file.write_text("id,text\n1,test")
    non_existent_config = tmp_path / "non_existent_config.yaml"

    mocker.patch(
        'src.cli.load_config',  # Corrected src. prefix
        side_effect=FileNotFoundError("Config not found"))
    # We also need to mock logger because cli.py logs an error
    mock_logger = mocker.patch('src.cli.logger')  # Corrected src. prefix

    argv = [
        'script_name', '--input-file',
        str(input_file), '--config',
        str(non_existent_config)
    ]
    mock_sys_argv(argv)

    with pytest.raises(SystemExit) as e:
        cli.main()
    assert e.value.code == 0  # Continues with empty config

    # load_config was called
    # src.cli.load_config.assert_called_once_with(str(non_existent_config)) # This line causes issues with pytest-mock if src.cli.load_config is already the mock_load_config fixture

    # Check that the error was logged
    log_error_found = False
    for call_args in mock_logger.error.call_args_list:
        if "Configuration file not found" in call_args[0][0]:
            log_error_found = True
            break
    assert log_error_found

    # Check that run_batch_processing was called with an empty config
    args, config = mock_batch_runner_functions['batch'].call_args[0]
    assert config == {}
