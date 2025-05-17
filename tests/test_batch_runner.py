from pathlib import Path
from unittest.mock import MagicMock

import pytest

from batchgrader.batch_runner import process_file, run_batch_processing


@pytest.fixture
def input_file(tmp_path: Path) -> Path:
    """Fixture to create a dummy input file."""
    input_file = tmp_path / "dummy_input.csv"
    input_file.write_text("id,response\n1,Test\n")
    return input_file


@pytest.fixture
def input_dir(tmp_path: Path) -> Path:
    """Fixture to create a dummy input directory."""
    input_dir = tmp_path / "dummy_input_dir"
    input_dir.mkdir()
    dummy_file = input_dir / "dummy_input.csv"
    dummy_file.write_text("id,response\n1,Test\n")
    return input_dir


def test_run_batch_processing_single_file(mocker: MagicMock, input_file: Path,
                                          basic_config: dict) -> None:
    """Test successful batch processing of a single file."""
    mock_process_file = mocker.patch("batchgrader.batch_runner.process_file",
                                     return_value=True)
    mocker.patch("batchgrader.batch_runner.prune_logs_if_needed")

    mock_args = MagicMock()
    mock_args.input_file = str(input_file)
    mock_args.input_dir = None
    mock_args.output_dir = str(input_file.parent)
    mock_args.log_dir = None  # For get_log_dirs
    mock_args.reprocess = False  # For process_file via _process_file_with_config

    run_batch_processing(mock_args, basic_config)

    mock_process_file.assert_called_once_with(
        str(input_file),
        str(input_file.parent
            ),  # This is derived from args.output_dir in the call
        basic_config,
        mock_args,  # process_file now receives args
    )


def test_run_batch_processing_input_dir(mocker: MagicMock, input_dir: Path,
                                        basic_config: dict) -> None:
    """Test successful batch processing of an input directory."""
    mock_process_file = mocker.patch("batchgrader.batch_runner.process_file",
                                     return_value=True)
    mocker.patch("batchgrader.batch_runner.prune_logs_if_needed")

    mock_args = MagicMock()
    mock_args.input_file = None
    mock_args.input_dir = str(input_dir)
    mock_args.output_dir = str(input_dir)
    mock_args.log_dir = None
    mock_args.reprocess = False

    run_batch_processing(mock_args, basic_config)

    expected_input_file = str(input_dir / "dummy_input.csv")
    expected_output_dir = str(
        input_dir)  # This is derived from args.output_dir
    mock_process_file.assert_called_once_with(
        expected_input_file,
        expected_output_dir,
        basic_config,
        mock_args,  # process_file now receives args
    )


def test_process_file_success(mocker: MagicMock, input_file: Path,
                              basic_config: dict) -> None:
    """Test successful processing of a single file."""
    mock_process_file_wrapper = mocker.patch(
        "batchgrader.batch_runner.process_file_wrapper", return_value=True)
    mocker.patch(
        "batchgrader.batch_runner.load_system_prompt",
        return_value="Test System Prompt Content",
    )

    mock_encoder = mocker.Mock(name="encoder")
    mock_llm_client = mocker.patch("batchgrader.batch_runner.LLMClient")
    mock_llm_client.return_value.encoder = mock_encoder
    mocker.patch("tiktoken.encoding_for_model", return_value=mock_encoder)

    basic_config["response_field_name"] = "response"

    success = process_file(input_file, input_file.parent, basic_config)

    assert success is True
    mock_process_file_wrapper.assert_called_once_with(
        filepath=str(input_file),
        output_dir=str(input_file.parent),
        config=basic_config,
        system_prompt_content="Test System Prompt Content",
        response_field=basic_config["response_field_name"],
        encoder=mock_encoder,
        token_limit=basic_config["global_token_limit"],
        is_reprocessing_run=False,
    )


def test_process_file_failure(mocker: MagicMock, input_file: Path,
                              basic_config: dict) -> None:
    """Test failed processing of a single file."""
    _ = mocker.patch("batchgrader.batch_runner.process_file_wrapper",
                     return_value=False)
    mocker.patch(
        "batchgrader.batch_runner.load_system_prompt",
        return_value="Test System Prompt Content",
    )

    mock_encoder = mocker.Mock(name="encoder")
    mock_llm_client = mocker.patch("batchgrader.batch_runner.LLMClient")
    mock_llm_client.return_value.encoder = mock_encoder
    mocker.patch("tiktoken.encoding_for_model", return_value=mock_encoder)

    basic_config["response_field_name"] = "response"

    success = process_file(input_file, input_file.parent, basic_config)

    assert success is False
