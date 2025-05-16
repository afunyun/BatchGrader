import pytest
import pandas as pd
from pathlib import Path
import shutil
import os
import yaml
from unittest.mock import MagicMock, patch, Mock, call
import logging
import sys
import json
import tiktoken  # Import the mocked tiktoken module

# Import from the package
# from logger import logger as global_logger_for_tests # REMOVED
from src.llm_client import LLMClient  # Ensure explicit import
from src.config_loader import load_config  # Ensure explicit import
from src.logger import setup_logging  # Import setup_logging

# Import functions to be tested from batch_runner
from src.batch_runner import (
    process_file,
    run_batch_processing,
    run_count_mode,  # Ensure explicit import
    run_split_mode,
    print_token_cost_stats,
    print_token_cost_summary)

# Import relevant items from other modules for mocking or setup
from src.file_processor import process_file_wrapper, process_file_concurrently  # Ensure explicit import
from src.input_splitter import split_file_by_token_limit  # Ensure explicit import
from src.constants import DEFAULT_RESPONSE_FIELD, DEFAULT_GLOBAL_TOKEN_LIMIT, DEFAULT_SPLIT_TOKEN_LIMIT  # Ensure explicit import
from src.prompt_utils import load_system_prompt  # Ensure explicit import

Path("tests/logs").mkdir(parents=True, exist_ok=True)


@pytest.fixture(scope="session")
def global_logger_instance(tmp_path_factory):
    log_dir = tmp_path_factory.mktemp("test_br_logs")
    setup_logging(log_dir=log_dir, log_level=logging.DEBUG)
    return logging.getLogger(
        "test_batch_runner")  # Use a named logger for tests


@pytest.fixture(scope="session")
def test_config_continue_multi_chunk_failure_fixture(global_logger_instance,
                                                     tmp_path_factory):
    config_path = Path(
        __file__).parent / "config_continue_multi_chunk_failure.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    test_dir = tmp_path_factory.mktemp("test_multi_chunk_failure_br")
    input_dir = test_dir / "input"
    output_dir = test_dir / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    input_file = input_dir / "large_chunk_br.csv"
    test_data = [{
        "id": i,
        "text_content": f"Test text {i}" * 100
    } for i in range(100)]
    pd.DataFrame(test_data).to_csv(input_file, index=False)

    config['input_dir'] = str(input_dir)
    config['output_options'] = config.get('output_options', {})
    config['output_options']['output_base_dir'] = str(output_dir)
    config['logger'] = global_logger_instance

    return config


@pytest.fixture
def mock_args_input_file(tmp_path):
    """Provides mock argparse args for a single input file."""
    input_file = tmp_path / "input.csv"
    pd.DataFrame([{'id': 1, 'text': 'test'}]).to_csv(input_file, index=False)
    return MagicMock(
        input_file=str(input_file),
        input_dir=None,
        output_dir=str(tmp_path / "output_br"),
        config_file=None,
        log_dir=None,
        mode='batch',  # Default mode
        stats=False)


@pytest.fixture
def mock_args_input_dir(tmp_path):
    """Provides mock argparse args for an input directory."""
    input_dir = tmp_path / "input_dir_br"
    input_dir.mkdir()
    (input_dir / "file1.csv").write_text("id,text\n1,test1")
    (input_dir / "file2.jsonl").write_text('{"id": 2, "text": "test2"}\n')
    return MagicMock(input_file=None,
                     input_dir=str(input_dir),
                     output_dir=str(tmp_path / "output_br_dir"),
                     config_file=None,
                     log_dir=None,
                     mode='batch',
                     stats=False)


@pytest.fixture
def basic_config():
    """A basic config dictionary for tests."""
    return {
        "system_prompt": "Test prompt",
        "response_field_name": DEFAULT_RESPONSE_FIELD,
        "openai_model_name": "gpt-3.5-turbo",
        "global_token_limit": DEFAULT_GLOBAL_TOKEN_LIMIT,
        "split_token_limit": DEFAULT_SPLIT_TOKEN_LIMIT,
        "input_splitter_options": {},
        "output_options": {},
        "llm_client_options": {
            "api_key": "TEST_KEY"
        },
        "examples_dir": "tests/input/examples",  # Add missing examples_dir
        "system_prompt_template": "tests/config/test_system_prompt.txt"
    }


def test_run_batch_processing_single_file(mocker, mock_args_input_file,
                                          basic_config):
    mock_process_file = mocker.patch('src.batch_runner.process_file',
                                     return_value=True)
    mocker.patch(
        'src.batch_runner.prune_logs_if_needed')  # Mock to avoid side effects

    run_batch_processing(mock_args_input_file, basic_config)

    mock_process_file.assert_called_once_with(
        str(mock_args_input_file.input_file),
        str(Path(mock_args_input_file.output_dir)), basic_config)


def test_run_batch_processing_input_dir(mocker, mock_args_input_dir,
                                        basic_config):
    mock_process_file = mocker.patch('src.batch_runner.process_file',
                                     return_value=True)
    mocker.patch('src.batch_runner.prune_logs_if_needed')

    run_batch_processing(mock_args_input_dir, basic_config)

    assert mock_process_file.call_count == 2  # two files in mock_args_input_dir
    expected_calls = [
        call(str(Path(mock_args_input_dir.input_dir) / "file1.csv"),
             str(Path(mock_args_input_dir.output_dir)), basic_config),
        call(str(Path(mock_args_input_dir.input_dir) / "file2.jsonl"),
             str(Path(mock_args_input_dir.output_dir)), basic_config)
    ]
    mock_process_file.assert_has_calls(expected_calls, any_order=True)


def test_process_file(mocker, mock_args_input_file, basic_config):
    mock_file_wrapper = mocker.patch('src.batch_runner.process_file_wrapper',
                                     return_value=True)
    mocker.patch('src.batch_runner.load_system_prompt',
                 return_value="Test System Prompt Content")

    # Create a consistent encoder mock
    mock_encoder = mocker.Mock(name='encoder')
    mock_llm_client = mocker.patch('src.batch_runner.LLMClient')
    mock_llm_client.return_value.encoder = mock_encoder
    # Ensure the encoder is used consistently
    mocker.patch('tiktoken.encoding_for_model', return_value=mock_encoder)

    success = process_file(mock_args_input_file.input_file,
                           mock_args_input_file.output_dir, basic_config)

    assert success is True
    mock_file_wrapper.assert_called_once_with(
        filepath=str(mock_args_input_file.input_file),
        output_dir=str(mock_args_input_file.output_dir),
        config=basic_config,
        system_prompt_content="Test System Prompt Content",
        response_field=basic_config['response_field_name'],
        encoder=mock_encoder,
        token_limit=basic_config['global_token_limit'])


def test_run_count_mode_single_file(mocker, mock_args_input_file,
                                    basic_config):
    mock_args_input_file.mode = 'count'
    # Create a consistent DataFrame for testing
    test_df = pd.DataFrame([{'text': 'sample data'}])
    mocker.patch('src.batch_runner.load_data', return_value=test_df)
    mock_check_limits = mocker.patch('src.batch_runner.check_token_limits',
                                     return_value=(True, {
                                         'total': 10,
                                         'average': 10.0,
                                         'max': 10
                                     }))

    # Create a consistent encoder mock
    mock_encoder = mocker.Mock(name='encoder')
    mock_llm_client = mocker.patch('src.batch_runner.LLMClient')
    mock_llm_client.return_value.encoder = mock_encoder
    # Ensure the encoder is used consistently
    mocker.patch('tiktoken.encoding_for_model', return_value=mock_encoder)
    mocker.patch('src.batch_runner.load_system_prompt', return_value="SysPrompt")

    run_count_mode(mock_args_input_file, basic_config)

    # Verify check_token_limits was called with correct arguments
    assert mock_check_limits.call_count == 1
    args, kwargs = mock_check_limits.call_args

    # Check that the DataFrame passed to check_token_limits matches our test DataFrame
    pd.testing.assert_frame_equal(args[0], test_df)

    # Verify other arguments
    assert args[1] == "SysPrompt"  # system_prompt_content
    assert args[2] == basic_config['response_field_name']  # response_field
    assert args[3] == mock_encoder  # encoder


def test_run_split_mode_single_file(mocker, mock_args_input_file,
                                    basic_config):
    """Test split mode with a single input file."""
    # Mock the split_file_by_token_limit function
    mock_split = mocker.patch('src.batch_runner.split_file_by_token_limit')
    mock_split.return_value = (["chunk1.csv"], 100)

    # Mock the process_file_wrapper function
    mocker.patch('src.batch_runner.process_file_wrapper', return_value=True)

    # Mock the load_system_prompt function
    mocker.patch('src.batch_runner.load_system_prompt', return_value="Test prompt")

    # Mock the get_encoder function
    mocker.patch('src.batch_runner.get_encoder', return_value=Mock())

    # Run split mode
    mock_args_input_file.mode = 'split'
    run_split_mode(mock_args_input_file, basic_config)

    # Verify the split_file_by_token_limit was called
    mock_split.assert_called_once()


def test_process_file_error_handling(mocker, mock_args_input_file,
                                     basic_config):
    """Test error handling in process_file function."""
    # Test FileNotFoundError
    mocker.patch('pathlib.Path.exists', return_value=False)
    result = process_file(str(mock_args_input_file.input_file),
                          str(mock_args_input_file.output_dir), basic_config)
    assert result is False

    # Test ValueError
    mocker.patch('pathlib.Path.exists', return_value=True)
    mocker.patch('src.batch_runner.load_system_prompt',
                 side_effect=ValueError("Invalid prompt"))
    result = process_file(str(mock_args_input_file.input_file),
                          str(mock_args_input_file.output_dir), basic_config)
    assert result is False

    # Test unexpected exception
    mocker.patch('src.batch_runner.load_system_prompt',
                 side_effect=Exception("Unexpected error"))
    result = process_file(str(mock_args_input_file.input_file),
                          str(mock_args_input_file.output_dir), basic_config)
    assert result is False


def test_token_cost_stats(mocker, basic_config):
    """Test token cost statistics functions."""
    # Mock LLMClient
    mock_llm = MagicMock()
    mock_llm.api_key = "test_key"
    mocker.patch('src.batch_runner.LLMClient', return_value=mock_llm)

    # Test successful token stats retrieval
    mock_usage = {'total_tokens': 1000, 'estimated_cost': 0.02}
    mocker.patch('src.batch_runner.get_token_usage_for_day',
                 return_value=mock_usage)
    print_token_cost_stats(basic_config)

    # Test no token usage
    mocker.patch('src.batch_runner.get_token_usage_for_day', return_value=None)
    print_token_cost_stats(basic_config)

    # Test exception handling
    mocker.patch('src.batch_runner.get_token_usage_for_day',
                 side_effect=Exception("API Error"))
    print_token_cost_stats(basic_config)


def test_token_cost_summary(mocker, tmp_path, basic_config):
    """Test token cost summary function."""
    # Mock LLMClient
    mock_llm = MagicMock()
    mock_llm.api_key = "test_key"
    mocker.patch('src.batch_runner.LLMClient', return_value=mock_llm)

    # Create mock summary file
    summary_file = tmp_path / "token_summary.json"
    summary_data = {
        'total_tokens_all_time': 5000,
        'total_estimated_cost_all_time': 0.10
    }
    summary_file.write_text(json.dumps(summary_data))

    # Test with existing summary file
    print_token_cost_summary(str(summary_file), tmp_path, basic_config)

    # Test with non-existent file
    print_token_cost_summary(str(tmp_path / "nonexistent.json"), tmp_path,
                             basic_config)

    # Test with invalid summary data
    invalid_summary = tmp_path / "invalid_summary.json"
    invalid_summary.write_text("invalid json")
    print_token_cost_summary(str(invalid_summary), tmp_path, basic_config)


def test_run_batch_processing_error_handling(mocker, mock_args_input_file,
                                             basic_config, tmp_path):
    """Test error handling in run_batch_processing."""
    # Mock log directory
    log_dir = tmp_path / "logs"
    archive_dir = log_dir / "archive"
    mocker.patch('src.batch_runner.get_log_dirs',
                 return_value=(log_dir, archive_dir))
    mocker.patch('src.batch_runner.prune_logs_if_needed', return_value=None)

    # Test input file doesn't exist
    mock_exists = mocker.patch('pathlib.Path.exists')
    mock_exists.return_value = False
    run_batch_processing(mock_args_input_file, basic_config)

    # Test input directory doesn't exist
    mock_args_input_file.input_file = None
    mock_args_input_file.input_dir = "/nonexistent/dir"
    run_batch_processing(mock_args_input_file, basic_config)

    # Test output directory creation failure
    def mock_mkdir(*args, **kwargs):
        path = kwargs.get('path', None) if kwargs else (args[0] if args else None)
        if path and 'output' in str(path):
            raise PermissionError("Access denied")

    mock_exists.return_value = True
    mocker.patch('pathlib.Path.mkdir', side_effect=mock_mkdir)
    run_batch_processing(mock_args_input_file, basic_config)

    # Test process_file failure
    mocker.patch('pathlib.Path.mkdir', return_value=None)
    mock_process = mocker.patch('src.batch_runner.process_file')
    mock_process.return_value = False
    run_batch_processing(mock_args_input_file, basic_config)


def test_run_batch_processing_with_continue_on_failure(mocker, tmp_path,
                                                       basic_config):
    """Test batch processing with continue_on_failure option."""
    # Setup test files
    input_dir = tmp_path / "input_dir"
    input_dir.mkdir()
    (input_dir / "file1.csv").write_text("id,text\n1,test1")
    (input_dir / "file2.csv").write_text("id,text\n2,test2")

    mock_args = MagicMock(input_file=None,
                          input_dir=str(input_dir),
                          output_dir=str(tmp_path / "output"),
                          config_file=None,
                          log_dir=None,
                          mode='batch')

    # Mock process_file to fail for first file but succeed for second
    process_file_mock = mocker.patch('src.batch_runner.process_file')
    process_file_mock.side_effect = [False, True]

    # Enable continue_on_failure in config
    config = basic_config.copy()
    config['continue_on_failure'] = True

    run_batch_processing(mock_args, config)

    # Verify both files were processed despite first failure
    assert process_file_mock.call_count == 2


def test_run_count_mode_error_handling(mocker, mock_args_input_file,
                                       basic_config):
    """Test error handling in run_count_mode."""
    mock_args_input_file.mode = 'count'

    # Test file not found
    mocker.patch('pathlib.Path.exists', return_value=False)
    run_count_mode(mock_args_input_file, basic_config)

    # Test encoder initialization failure
    mocker.patch('pathlib.Path.exists', return_value=True)
    mocker.patch('src.batch_runner.get_encoder', return_value=None)
    run_count_mode(mock_args_input_file, basic_config)

    # Test token counting failure
    mock_encoder = MagicMock()
    mock_encoder.encode.side_effect = Exception("Encoding error")
    mocker.patch('src.batch_runner.get_encoder', return_value=mock_encoder)
    run_count_mode(mock_args_input_file, basic_config)


def test_run_split_mode_error_handling(mocker, mock_args_input_file,
                                       basic_config):
    """Test error handling in run_split_mode."""
    mock_args_input_file.mode = 'split'

    # Test file not found
    mocker.patch('pathlib.Path.exists', return_value=False)
    run_split_mode(mock_args_input_file, basic_config)

    # Test split failure
    mocker.patch('pathlib.Path.exists', return_value=True)
    mocker.patch('src.batch_runner.split_file_by_token_limit',
                 side_effect=Exception("Split error"))
    run_split_mode(mock_args_input_file, basic_config)

    # Test output directory creation failure
    mocker.patch('src.batch_runner.split_file_by_token_limit', return_value=True)
    mocker.patch('pathlib.Path.mkdir',
                 side_effect=PermissionError("Access denied"))
    run_split_mode(mock_args_input_file, basic_config)
