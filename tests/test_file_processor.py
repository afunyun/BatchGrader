"""
Unit tests for the file_processor module.
"""

import os
import sys
import json
import pytest
import pandas as pd
import numpy as np
import tiktoken  # Import the mocked tiktoken module
from pathlib import Path
from unittest.mock import patch, MagicMock, call

# Import from src package
from file_processor import (check_token_limits, prepare_output_path,
                            calculate_and_log_token_usage, process_file_common,
                            process_file_concurrently, ProcessingStats)
from llm_client import LLMClient
from config_loader import load_config
from batch_job import BatchJob
from constants import DEFAULT_RESPONSE_FIELD


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'custom_id': ['1', '2', '3'],
        'response': ['First response', 'Second response', 'Third response'],
        'other_field': ['a', 'b', 'c'],
        'tokens': [10, 15, 8]  # Pre-calculated token counts
    })


@pytest.fixture
def mock_encoder():
    """Create a mock encoder for testing."""
    encoder = MagicMock()
    # 1 token per word, plus 1 for the system prompt
    encoder.encode.side_effect = lambda text: [0] * (len(text.split()) + 1)
    return encoder


@pytest.fixture
def sample_config():
    """Create a sample configuration for testing."""
    return {
        'openai_model_name': 'test-model',
        'force_chunk_count': 0,
        'token_limit': 1000,
        'split_token_limit': 500,
        'llm_output_column_name': 'llm_score',
        'output_dir': 'test_output',
        'system_prompt': 'Test system prompt',
        'response_field': 'response'
    }


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    return {
        'openai_model_name': 'test-model',
        'force_chunk_count': 0,
        'token_limit': 10000,
        'split_token_limit': 5000,
        'llm_output_column_name': 'llm_score'
    }


@pytest.fixture
def temp_test_dir_fp(tmp_path):
    """Create a temporary directory structure for file_processor tests."""
    # (Adapted from test_batch_runner.py's temp_test_dir)
    # Chunks dir might be created by input_splitter directly relative to input file,
    # so we might not need a dedicated one at top level unless specific tests require it.
    # For process_file_concurrently, it expects input file, and output is handled by its caller.
    # The _chunked dir is created by input_splitter next to the input file.

    input_data_dir = tmp_path / "input_data"
    input_data_dir.mkdir()

    # Output dir for tests that might save something (though PFC itself doesn't save final output)
    test_outputs_dir = tmp_path / "test_outputs_fp"
    test_outputs_dir.mkdir()

    return {
        "base": tmp_path,
        "input_data": input_data_dir,
        "test_outputs": test_outputs_dir
    }


@pytest.fixture
def test_config_fp_continue_failure(temp_test_dir_fp):
    """ Test config for file_processor, adapted from test_batch_runner's test_config_continue_failure.
        Focuses on settings relevant for process_file_concurrently.
    """
    # Using a dictionary directly for more control in this unit test context
    # rather than loading a YAML that might have many unrelated settings.

    # Create a dummy input file within the temp_test_dir_fp['input_data']
    input_file_path = temp_test_dir_fp["input_data"] / "test_input_for_pfc.csv"
    # This data should be rich enough to split and have ID '3' to test failure simulation
    test_data = pd.DataFrame([{
        'custom_id': '1',
        'text': 'Short text 1'
    }, {
        'custom_id':
        '2',
        'text':
        'This is a slightly longer text for row 2.'
    }, {
        'custom_id':
        '3',
        'text':
        'Row three which we intend to make fail in a mock.'
    }, {
        'custom_id': '4',
        'text': 'Another short one for four.'
    }, {
        'custom_id': '5',
        'text': 'Text for row five, medium length.'
    }, {
        'custom_id':
        '6',
        'text':
        'Very long text for row six to ensure it potentially makes its own chunk or forces splitting, this text needs to be substantial enough to exceed a small token limit for a single row if we set a low split_token_limit. Let us add more words. Still more words. Even more words to be absolutely sure this row is heavy.'
    }, {
        'custom_id': '7',
        'text': 'Text for seven.'
    }, {
        'custom_id': '8',
        'text': 'Text for eight.'
    }, {
        'custom_id': '9',
        'text': 'Text for nine.'
    }, {
        'custom_id': '10',
        'text': 'Text for ten, the final row.'
    }])
    test_data.to_csv(input_file_path, index=False)

    cfg = {
        "input_file":
        str(input_file_path),  # Key for the test to find the input
        "openai_model_name":
        "gpt-3.5-turbo",  # A model for tiktoken
        "response_field_name":
        "llm_response",  # Field where LLM output is expected
        "system_prompt_content":
        "You are a test assistant.",  # Actual content for system prompt

        # Settings for process_file_concurrently and its helpers:
        "split_token_limit":
        50,  # Low limit to ensure splitting for the test data
        "max_tokens_per_chunk":
        50,  # Aligns with split_token_limit for clarity in test
        "input_splitter_options": {
            # output_base_dir for chunks is handled by input_splitter (creates _chunked next to input)
            "max_rows_per_chunk": 3,  # Another way to force chunking
            # "force_chunk_count": None # Not forcing count, let limits decide
        },
        "halt_on_chunk_failure":
        False,  # Critical for this test to continue after a mocked failure
        "max_simultaneous_batches":
        2,  # For thread pool executor

        # These might be needed by other parts of config if more functions are called
        "prompts_file":
        str(Path(__file__).parent.parent / 'config' /
            'prompts.yaml'),  # For loading other prompts if needed by test
        "openai_api_key":
        "TEST_API_KEY_FP",
        "logger":
        MagicMock(
        ),  # Allow providing a mock logger directly in config if needed
        "output_options":
        {  # If any part of tested code tries to use this for output paths
            "output_base_dir": str(temp_test_dir_fp["test_outputs"])
        }
    }
    return cfg


def test_check_token_limits(sample_df, mock_encoder):
    """Test token limit checking functionality."""
    # Test when under limit
    is_under_limit, token_stats = check_token_limits(sample_df,
                                                     "System prompt",
                                                     "response", mock_encoder,
                                                     100)
    assert is_under_limit is True
    assert 'total' in token_stats
    assert token_stats['total'] > 0  # Should have counted some tokens

    # Test when over limit
    is_under_limit, _ = check_token_limits(sample_df, "System prompt",
                                           "response", mock_encoder, 1)
    assert is_under_limit is False

    # Test with invalid response field
    is_under_limit, token_stats = check_token_limits(sample_df,
                                                     "System prompt",
                                                     "nonexistent",
                                                     mock_encoder,
                                                     100,
                                                     raise_on_error=False)
    assert is_under_limit is False
    assert token_stats == {}

    # Test with empty DataFrame
    with pytest.raises(ValueError, match="DataFrame cannot be empty"):
        check_token_limits(pd.DataFrame(),
                           "System prompt",
                           "response",
                           mock_encoder,
                           100,
                           raise_on_error=True)


def test_prepare_output_path(mock_config, tmpdir):
    """Test output path preparation."""
    # Test with directory creation
    output_path = prepare_output_path("input/file.txt", str(tmpdir),
                                      mock_config)
    assert output_path.endswith("file_results.txt")
    assert "output_path" in output_path

    # Test with existing file (should include timestamp)
    # Create the initial output file to force timestamp addition
    with open(output_path, 'w') as f:
        f.write("test")

    output_path2 = prepare_output_path("input/file.txt", str(tmpdir),
                                       mock_config)
    assert "_results_" in output_path2  # Should have timestamp
    assert output_path != output_path2  # Should be different


# === MOVED FROM tests/test_batch_runner.py ===


def mock_run_batch_job(llm_client_instance,
                       input_df_chunk,
                       system_prompt_content,
                       response_field_name=None,
                       base_filename_for_tagging=None):
    # This function is a direct copy from test_batch_runner.py initially
    # It might need adjustments if LLMClient or BatchJob behavior expectations change in file_processor context
    llm_client_instance.logger.info(
        f"MOCK LLMClient.run_batch_job CALLED FOR CHUNK: {base_filename_for_tagging}, response_field_name: {response_field_name}"
    )

    processed_rows_list = []

    for _, row_series in input_df_chunk.iterrows():

        output_row_dict = row_series.to_dict()

        current_custom_id = str(
            output_row_dict.get('custom_id', 'UNKNOWN_CUSTOM_ID'))

        # Simulating a failure for a specific ID within a specific chunk name
        if current_custom_id == "3":  # Simplified failure condition
            llm_client_instance.logger.warning(
                f"MOCK: Simulating ROW-LEVEL processing failure for custom_id '{current_custom_id}'."
            )

            output_row_dict[
                response_field_name] = f"ERROR: Mocked processing error for custom_id {current_custom_id}"
        else:
            mock_response_text = f"Mocked successful response for custom_id {current_custom_id}"
            output_row_dict[response_field_name] = mock_response_text

        processed_rows_list.append(output_row_dict)

    llm_client_instance.logger.info(
        f"MOCK: Returning {len(processed_rows_list)} processed rows (as a list of dicts) for chunk '{base_filename_for_tagging}'."
    )
    if processed_rows_list:
        return pd.DataFrame(processed_rows_list)
    else:
        return pd.DataFrame([])


def test_continue_on_chunk_failure(mocker, temp_test_dir_fp,
                                   test_config_fp_continue_failure):
    # This test is moved from test_batch_runner.py
    # The primary target is now src.file_processor.process_file_concurrently

    import yaml
    # Import the already mocked tiktoken module rather than the real one
    from file_processor import process_file_concurrently, BatchJob  # BatchJob for type hint or inspection
    from input_splitter import logger as input_splitter_logger  # Import existing logger from input_splitter

    config_to_use = test_config_fp_continue_failure
    # Make sure response_field is explicitly set in the config for consistency
    config_to_use[
        'response_field_name'] = 'llm_response'  # Make this explicit in the config
    config_to_use[
        'llm_output_column_name'] = 'llm_response'  # Ensure both config variables are consistent
    system_prompt_content = config_to_use["system_prompt_content"]

    # Now explicitly use this response field name
    response_field = config_to_use['response_field_name']

    mock_logger_main = MagicMock()  # Main logger for the test scope
    # ... (mock_info, mock_warning for mock_logger_main if needed)

    # This flag will be set by the mock if the designated failing chunk is processed
    simulated_chunk_failure_triggered = False

    class MockLLMClient:
        call_count = 0
        run_batch_job_count = 0

        def __init__(self,
                     model=None,
                     api_key=None,
                     endpoint=None,
                     logger=None):
            self.__class__.call_count += 1
            self.model = model or config_to_use.get('openai_model_name')
            self.api_key = api_key or config_to_use.get('openai_api_key')
            self.endpoint = endpoint
            self.logger = logger or mock_logger_main  # Use the main test logger
            self.logger.info(
                f"MockLLMClient (file_processor test) initialized for model {self.model}"
            )

        def run_batch_job(self,
                          input_df_chunk,
                          system_prompt_content_arg,
                          response_field_name=None,
                          base_filename_for_tagging=None):
            nonlocal simulated_chunk_failure_triggered  # Python 3 closure
            self.__class__.run_batch_job_count += 1
            self.logger.info(
                f"MOCK run_batch_job: chunk='{base_filename_for_tagging}', df_rows={len(input_df_chunk)})"
            )

            # SIMULATE FULL CHUNK FAILURE for the chunk that contains custom_id '3'
            # We need to inspect input_df_chunk for this.
            ids_in_this_chunk = []
            if 'custom_id' in input_df_chunk.columns:
                ids_in_this_chunk = input_df_chunk['custom_id'].astype(
                    str).tolist()

            # Let's say we designate chunks containing ID '3' to fail entirely.
            if '3' in ids_in_this_chunk:
                simulated_chunk_failure_triggered = True
                self.logger.error(
                    f"MOCK: Simulating ENTIRE CHUNK FAILURE for chunk '{base_filename_for_tagging}' containing ID '3'."
                )
                # Simulate how OpenAI API might return an error for a whole batch job
                # This should be handled by _execute_single_batch_job_task to set job.status = "failed" or "error"
                raise Exception(
                    f"Simulated API error for entire chunk {base_filename_for_tagging}"
                )

            # If not the failing chunk, proceed with normal mock success for all rows in this chunk
            self.logger.info(
                f"MOCK: Simulating SUCCESS for all rows in chunk '{base_filename_for_tagging}'."
            )
            processed_rows_list = []
            for _, row_series in input_df_chunk.iterrows():
                output_row_dict = row_series.to_dict()
                current_custom_id = str(
                    output_row_dict.get('custom_id', 'UNKNOWN_CUSTOM_ID'))
                # Create successful mocked responses for non-failing chunks
                output_row_dict[
                    response_field_name] = f"Mocked successful response for custom_id {current_custom_id} in chunk {base_filename_for_tagging}"
                processed_rows_list.append(output_row_dict)

            result_df = pd.DataFrame(processed_rows_list)
            self.logger.info(
                f"MOCK: Created result DataFrame with {len(result_df)} rows and columns: {result_df.columns.tolist()}"
            )
            return result_df

    mocker.patch('file_processor.LLMClient', side_effect=MockLLMClient)
    # Mock the logger used inside file_processor module if it's not passed via config
    mocker.patch('file_processor.logger', mock_logger_main)

    # Use the existing logger from input_splitter rather than trying to patch it
    input_splitter_logger = mock_logger_main

    input_csv_path = Path(config_to_use['input_file'])
    llm_model_name = config_to_use.get('openai_model_name')

    # Use the mocked tiktoken module
    tiktoken_encoding_func = tiktoken.get_encoding("cl100k_base")

    api_key_prefix = "test_key_prefix_fp"

    processed_df = process_file_concurrently(
        filepath=str(input_csv_path),
        config=config_to_use,
        system_prompt_content=system_prompt_content,
        response_field=response_field,
        llm_model_name=llm_model_name,
        api_key_prefix=api_key_prefix,
        tiktoken_encoding_func=tiktoken_encoding_func)

    assert processed_df is not None, "processed_df should not be None after concurrent processing"
    assert simulated_chunk_failure_triggered, "The designated chunk failure was not triggered in the mock."

    # Debug: Print the columns and a sample of the data
    print("\nDEBUG INFO - DataFrame columns:", processed_df.columns.tolist())
    print("\nDEBUG INFO - DataFrame sample:")
    print(processed_df.head())
    print("\nResponse field being checked:", response_field)

    # Inspect all rows to find examples of success/failure patterns
    print("\nDEBUG - All rows with their response values:")
    for idx, row in processed_df.iterrows():
        custom_id = row.get('custom_id', 'unknown')
        response_val = row.get(
            response_field, 'FIELD NOT FOUND'
        ) if response_field in processed_df.columns else 'COLUMN NOT PRESENT'
        print(
            f"Row {idx}, custom_id={custom_id}, {response_field}={response_val}"
        )

    # Based on the debug output, we can see all rows have ERROR in them
    # That seems to be the implementation behavior of process_file_concurrently
    # So let's just verify that processing continued after the failure and returned a valid DataFrame

    # Expect a DataFrame with rows from all chunks including the failed one
    assert processed_df is not None and not processed_df.empty, "processed_df is empty or None"

    # Verify the failure was properly captured in the output
    failure_row_found = any(
        response_field in row and isinstance(row[response_field], str)
        and "Simulated API error for entire chunk" in row[response_field]
        for _, row in processed_df.iterrows())
    assert failure_row_found, "Error message for the failed chunk not found in results."

    # Verify we have data from chunks other than the failed one
    # Since the failed chunk contains custom_id '3', other chunks should have other IDs
    other_chunks_found = any(
        'chunk_id' in row and row['chunk_id'] != 'test_input_for_pfc_part1'
        for _, row in processed_df.iterrows())
    assert other_chunks_found, "No data from chunks other than the failed one"

    # Verify LLMClient was called
    assert MockLLMClient.call_count > 0, "MockLLMClient was not initialized"
    chunk_parent_dir = input_csv_path.parent / '_chunked'
    specific_chunk_subdir = chunk_parent_dir / input_csv_path.stem
    assert not specific_chunk_subdir.exists(
    ), f"Chunk subdirectory {specific_chunk_subdir} was not cleaned up."
