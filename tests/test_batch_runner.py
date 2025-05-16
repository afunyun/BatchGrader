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
from logger import logger as global_logger_for_tests
from llm_client import LLMClient
from config_loader import load_config

# Import functions to be tested from batch_runner
from batch_runner import (process_file, run_batch_processing, run_count_mode,
                          run_split_mode, print_token_cost_stats,
                          print_token_cost_summary)

# Import relevant items from other modules for mocking or setup
from file_processor import process_file_wrapper, process_file_concurrently
from input_splitter import split_file_by_token_limit
from constants import DEFAULT_RESPONSE_FIELD, DEFAULT_GLOBAL_TOKEN_LIMIT, DEFAULT_SPLIT_TOKEN_LIMIT
from prompt_utils import load_system_prompt

Path("tests/logs").mkdir(parents=True, exist_ok=True)


@pytest.fixture(scope="session")
def global_logger_instance():
    return global_logger_for_tests


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
    mock_process_file = mocker.patch('batch_runner.process_file',
                                     return_value=True)
    mocker.patch(
        'batch_runner.prune_logs_if_needed')  # Mock to avoid side effects

    run_batch_processing(mock_args_input_file, basic_config)

    mock_process_file.assert_called_once_with(
        str(mock_args_input_file.input_file),
        str(Path(mock_args_input_file.output_dir)), basic_config)


def test_run_batch_processing_input_dir(mocker, mock_args_input_dir,
                                        basic_config):
    mock_process_file = mocker.patch('batch_runner.process_file',
                                     return_value=True)
    mocker.patch('batch_runner.prune_logs_if_needed')

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
    mock_file_wrapper = mocker.patch('batch_runner.process_file_wrapper',
                                     return_value=True)
    mocker.patch('batch_runner.load_system_prompt',
                 return_value="Test System Prompt Content")
    # Mock LLMClient used to get encoder in process_file
    mock_llm_client_instance = MagicMock()
    # Use the mocked tiktoken module
    mock_llm_client_instance.encoder = tiktoken.get_encoding("cl100k_base")
    mocker.patch('batch_runner.LLMClient',
                 return_value=mock_llm_client_instance)

    success = process_file(mock_args_input_file.input_file,
                           mock_args_input_file.output_dir, basic_config)

    assert success is True
    mock_file_wrapper.assert_called_once_with(
        filepath=str(mock_args_input_file.input_file),
        output_dir=str(mock_args_input_file.output_dir),
        config=basic_config,
        system_prompt_content="Test System Prompt Content",
        response_field=basic_config['response_field_name'],
        encoder=mock_llm_client_instance.encoder,
        token_limit=basic_config['global_token_limit'])


def test_run_count_mode_single_file(mocker, mock_args_input_file,
                                    basic_config):
    mock_args_input_file.mode = 'count'
    # Create a consistent DataFrame for testing
    test_df = pd.DataFrame([{'text': 'sample data'}])
    mocker.patch('batch_runner.load_data', return_value=test_df)
    mock_check_limits = mocker.patch('batch_runner.check_token_limits',
                                     return_value=(True, {
                                         'total': 10,
                                         'average': 10.0,
                                         'max': 10
                                     }))

    mock_llm_client_instance = MagicMock()
    # Use the mocked tiktoken module
    mock_llm_client_instance.encoder = tiktoken.get_encoding("cl100k_base")
    mocker.patch('batch_runner.LLMClient',
                 return_value=mock_llm_client_instance)
    mocker.patch('batch_runner.load_system_prompt', return_value="SysPrompt")

    run_count_mode(mock_args_input_file, basic_config)

    # Use any_call instead of assert_called_once_with for DataFrame comparisons
    mock_check_limits.assert_any_call(
        mocker.ANY,  # Use ANY for DataFrame comparison
        "SysPrompt",
        basic_config['response_field_name'],
        mock_llm_client_instance.encoder,
        token_limit=float('inf'))

    # Verify the DataFrame was passed correctly by checking the first argument
    args, _ = mock_check_limits.call_args
    assert isinstance(args[0], pd.DataFrame)
    assert args[0].equals(test_df)


def test_run_split_mode_single_file(mocker, mock_args_input_file,
                                    basic_config):
    mock_args_input_file.mode = 'split'
    mock_split_file = mocker.patch('batch_runner.split_file_by_token_limit',
                                   return_value=(["chunk1.csv"], 100))

    mock_llm_client_instance = MagicMock()
    # Use the mocked tiktoken module
    mock_llm_client_instance.encoder = tiktoken.get_encoding("cl100k_base")
    mocker.patch('batch_runner.LLMClient',
                 return_value=mock_llm_client_instance)
    mocker.patch('batch_runner.load_system_prompt', return_value="SysPrompt")

    run_split_mode(mock_args_input_file, basic_config)

    # Check that split_file_by_token_limit was called.
    # Verifying all its args is complex because of the inner _row_token_counter_for_splitter.
    # A basic check that it's called is a good start.
    mock_split_file.assert_called_once()
    # Can add more specific arg checks if needed, e.g., for input_path, token_limit
    call_args = mock_split_file.call_args[1]  # Get kwargs
    assert call_args['input_path'] == str(mock_args_input_file.input_file)
    assert call_args['token_limit'] == basic_config['split_token_limit']
