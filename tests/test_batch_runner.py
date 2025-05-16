import pytest
import pandas as pd
from pathlib import Path
import shutil
import os
import tiktoken
import yaml
from unittest.mock import MagicMock, patch, Mock
import logging
import sys
import json
from src.logger import logger as global_logger_for_tests
from src.batch_runner import process_file_concurrently
from src.llm_client import LLMClient
from src.config_loader import load_config
import tiktoken

Path("tests/logs").mkdir(parents=True, exist_ok=True)

@pytest.fixture(scope="session")
def global_logger_instance():
    return global_logger_for_tests

@pytest.fixture
def temp_test_dir(tmp_path):
    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir()
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    return {"base": tmp_path, "chunks": chunk_dir, "outputs": output_dir}

@pytest.fixture
def test_config_continue_failure(temp_test_dir):
    config_path = Path(__file__).parent / "test_config_continue_on_failure.yaml"
    cfg = load_config(config_path)
    
    # Update paths to use the temp directory
    input_file = Path(cfg["input_file"])
    if not input_file.is_absolute():
        input_file = Path(__file__).parent.parent / input_file
    cfg["input_file"] = str(input_file)
    
    # Ensure the input file exists
    if not Path(cfg["input_file"]).exists():
        # Create a test input file if it doesn't exist
        test_data = [{"id": i, "text": f"Test text {i}"} for i in range(10)]
        Path(cfg["input_file"]).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(test_data).to_csv(cfg["input_file"], index=False)
    
    # Set up output directory
    output_dir = Path(temp_test_dir["outputs"])
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg["output_file"] = str(output_dir / "output.csv")
    
    # Add missing configuration sections
    cfg["input_splitter_options"] = {"output_base_dir": str(temp_test_dir["chunks"])}
    cfg["output_options"] = {"output_base_dir": str(temp_test_dir["outputs"])}
    cfg["llm_client_options"] = {"api_key": "TEST_API_KEY"}
    
    return cfg

@pytest.fixture(scope="session")
def test_config_continue_multi_chunk_failure_fixture(global_logger_instance, tmp_path_factory):
    config_path = Path(__file__).parent / "config_continue_multi_chunk_failure.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set up test directories
    test_dir = tmp_path_factory.mktemp("test_multi_chunk_failure")
    input_dir = test_dir / "input"
    output_dir = test_dir / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    
    # Create a test input file
    input_file = input_dir / "large_chunk.csv"
    test_data = [{"id": i, "text_content": f"Test text {i}" * 100} for i in range(100)]
    pd.DataFrame(test_data).to_csv(input_file, index=False)
    
    # Update config with test paths
    config['input_dir'] = str(input_dir)
    config['output_options']['output_base_dir'] = str(output_dir)
    config['logger'] = global_logger_instance
    
    return config

def mock_run_batch_job(llm_client_instance, input_df_chunk, system_prompt_content, response_field_name=None, base_filename_for_tagging=None):
    llm_client_instance.logger.info(
        f"MOCK LLMClient.run_batch_job CALLED FOR CHUNK: {base_filename_for_tagging}, response_field_name: {response_field_name}"
    )
    
    processed_rows_list = []

    for _, row_series in input_df_chunk.iterrows():
        
        output_row_dict = row_series.to_dict()
        
        current_custom_id = str(output_row_dict.get('custom_id', 'UNKNOWN_CUSTOM_ID'))
        
        if current_custom_id == "3" and base_filename_for_tagging and "chunk_with_failure" in base_filename_for_tagging:
            llm_client_instance.logger.warning(
                f"MOCK: Simulating ROW-LEVEL processing failure for custom_id '{current_custom_id}' in chunk '{base_filename_for_tagging}'."
            )
            
            output_row_dict[response_field_name] = f"ERROR: Mocked processing error for custom_id {current_custom_id}"
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

def test_continue_on_chunk_failure(mocker, temp_test_dir, test_config_continue_failure):
    # Load the prompts from the prompts file
    prompts_file = Path(test_config_continue_failure.get('prompts_file', '../config/prompts.yaml'))
    if not prompts_file.exists():
        # Try absolute path from project root
        prompts_file = Path(__file__).parent.parent / 'config' / 'prompts.yaml'
    with open(prompts_file, 'r', encoding='utf-8') as f:
        prompts = yaml.safe_load(f)
    
    # Create a mock logger
    mock_logger = mocker.MagicMock()
    
    # Configure the mock logger's info method to print to console for test visibility
    def mock_info(msg, *args, **kwargs):
        print(f"[INFO] {msg % args if args else msg}")
    mock_logger.info = mock_info
    
    # Configure the mock logger's warning method
    def mock_warning(msg, *args, **kwargs):
        print(f"[WARNING] {msg % args if args else msg}")
    mock_logger.warning = mock_warning
    
    # Create a mock LLMClient class
    class MockLLMClient:
        call_count = 0
        run_batch_job_count = 0
            
        def __init__(self, model=None, api_key=None, endpoint=None, logger=None):
            self.__class__.call_count += 1
            self.model = model or test_config_continue_failure.get('openai_model_name')
            self.api_key = api_key or test_config_continue_failure.get('openai_api_key', "TEST_API_KEY_INIT")
            self.endpoint = endpoint
            self.logger = logger or mock_logger
            self.logger.info(f"Mock LLMClient initialized for model {self.model}")
    
        def run_batch_job(self, input_df_chunk, system_prompt_content, response_field_name=None, base_filename_for_tagging=None):
            self.__class__.run_batch_job_count += 1
            return mock_run_batch_job(self, input_df_chunk, system_prompt_content, response_field_name, base_filename_for_tagging)
    
    # Patch the LLMClient with our mock
    mocker.patch('src.batch_runner.LLMClient', side_effect=MockLLMClient)
    
    input_csv_path = Path(test_config_continue_failure['input_file'])
    # Use the batch evaluation prompt from the prompts file
    system_prompt_content = prompts['batch_evaluation_prompt']
    response_field = 'response'  # Default response field for the test
    llm_model_name = test_config_continue_failure.get('openai_model_name')
    
    try:
        tiktoken_encoding_func = tiktoken.encoding_for_model(llm_model_name)
    except KeyError:
        tiktoken_encoding_func = tiktoken.get_encoding("cl100k_base")
    api_key_prefix = "test_key_prefix"

    processed_df = process_file_concurrently(
        filepath=input_csv_path,
        config=test_config_continue_failure,
        system_prompt_content=system_prompt_content,
        response_field=response_field,
        llm_model_name=llm_model_name,
        api_key_prefix=api_key_prefix,
        tiktoken_encoding_func=tiktoken_encoding_func
    )

    assert processed_df is not None
    assert len(processed_df) == 5

    expected_results_by_original_id = {
        "1": "Mocked successful response for custom_id 1",
        "2": "Mocked successful response for custom_id 2",
        "3": "ERROR: Mocked processing error for custom_id 3",
        "4": "Mocked successful response for custom_id 4",
        "5": "Mocked successful response for custom_id 5",
    }
    
    assert 'id' in processed_df.columns, "Original 'id' column must be in processed_df"
    assert response_field in processed_df.columns, f"'{response_field}' column must be in processed_df"
    for _, row in processed_df.iterrows():
        original_id = str(row['id'])
        assert row[response_field] == expected_results_by_original_id[original_id]
        # Verify that the LLMClient was initialized and run_batch_job was called
        assert MockLLMClient.call_count > 0
        assert MockLLMClient.run_batch_job_count > 0
        
        # Handle output file name generation
        output_options = test_config_continue_failure.get("output_options", {})
        if "output_filename_template" in output_options:
            output_file_name = output_options["output_filename_template"].format(original_filename=input_csv_path.stem)
        else:
            output_file_name = f"{input_csv_path.stem}_output.csv"
    expected_output_file = temp_test_dir["outputs"] / output_file_name
    
    if processed_df is not None:
        expected_output_file.parent.mkdir(parents=True, exist_ok=True)
        processed_df.to_csv(expected_output_file, index=False)
        
    assert expected_output_file.exists(), f"Output file {expected_output_file} was not created."
    df_output = pd.read_csv(expected_output_file)
    assert len(df_output) == 5
    
    for _, row in df_output.iterrows():
        original_id = str(row['id'])
        assert row[response_field] == expected_results_by_original_id[original_id]

def test_continue_after_full_chunk_failure_when_halt_is_false(
    mocker, 
    temp_test_dir, 
    test_config_continue_multi_chunk_failure_fixture, 
    global_logger_instance
):
    # Create a mock logger
    mock_logger = mocker.MagicMock()
    
    # Configure the mock logger's methods
    def mock_info(msg, *args, **kwargs):
        print(f"[INFO] {msg % args if args else msg}")
    mock_logger.info = mock_info
    
    def mock_warning(msg, *args, **kwargs):
        print(f"[WARNING] {msg % args if args else msg}")
    mock_logger.warning = mock_warning
    
    def mock_error(msg, *args, **kwargs):
        print(f"[ERROR] {msg % args if args else msg}")
    mock_logger.error = mock_error
    
    config = test_config_continue_multi_chunk_failure_fixture
    input_csv_path = Path(config['input_dir']) / "large_chunk.csv" 

    ids_in_failed_chunk = [] 
    
    # Create a mock LLMClient class for this test
    class MockLLMClient:
        call_count = 0
            
        def __init__(self, model=None, api_key=None, endpoint=None, logger=None):
            MockLLMClient.call_count += 1
            self.model = model or config['llm_client_options'].get('model')
            self.api_key = api_key or config['llm_client_options'].get('api_key', "TEST_API_KEY_INIT")
            self.endpoint = endpoint
            self.logger = logger or mock_logger
            self.logger.info(f"Mock LLMClient initialized for model {self.model}")

        def run_batch_job(self, input_df_chunk, system_prompt_content, response_field_name=None, base_filename_for_tagging=None):
            is_designated_failing_chunk = False

            if base_filename_for_tagging and "part1" in base_filename_for_tagging.rsplit('.', 1)[0]:
                is_designated_failing_chunk = True
            
            if is_designated_failing_chunk:
                self.logger.error(f"MOCK: Simulating APIError for chunk: {base_filename_for_tagging}")
                id_col_name = config['input_options'].get('id_column', 'id') 
                if id_col_name in input_df_chunk.columns: 
                    ids_in_failed_chunk.extend(input_df_chunk[id_col_name].astype(str).tolist())
                elif 'custom_id' in input_df_chunk.columns: 
                    ids_in_failed_chunk.extend(input_df_chunk['custom_id'].astype(str).tolist())
                raise Exception(f"Simulated API failure for chunk {base_filename_for_tagging}")

            processed_rows_list = []
            for _, row_series in input_df_chunk.iterrows():
                output_row_dict = row_series.to_dict()
                current_id = str(output_row_dict.get('id', 'UNKNOWN_ID')) 
                mock_response_text = f"Mocked successful response for id {current_id} in chunk {base_filename_for_tagging}"
                output_row_dict[response_field_name] = mock_response_text
                processed_rows_list.append(output_row_dict)
            
            self.logger.info(
                f"MOCK: Returning {len(processed_rows_list)} processed rows for successful chunk '{base_filename_for_tagging}'."
            )
            return pd.DataFrame(processed_rows_list) if processed_rows_list else pd.DataFrame([])
    
    # Patch the LLMClient with our mock
    mocker.patch('src.batch_runner.LLMClient', side_effect=MockLLMClient)
    
    def mock_llm_client_init_for_multi_chunk_test(self_client, model=None, api_key=None, endpoint=None, logger=None):
        self_client.model = model or config['llm_client_options'].get('model', config.get('openai_model_name'))
        self_client.api_key = api_key or config['llm_client_options'].get('api_key', "TEST_API_KEY_INIT_MULTI")
        self_client.endpoint = endpoint
        self_client.logger = logger or global_logger_instance
        self_client.logger.info(f"Mock LLMClient initialized for multi-chunk test, model {self_client.model}")
    mocked_llm_init = mocker.patch.object(LLMClient, '__init__', side_effect=mock_llm_client_init_for_multi_chunk_test, autospec=True)

    system_prompt_content = config['prompts']['system_prompt']
    response_field = config['response_field']
    llm_model_name = config.get('openai_model_name', config['llm_client_options'].get('model'))
    
    try:
        tiktoken_encoding_func = tiktoken.encoding_for_model(llm_model_name)
    except KeyError:
        tiktoken_encoding_func = tiktoken.get_encoding("cl100k_base")
    api_key_prefix = "test_key_prefix_multi_chunk"

    processed_df = process_file_concurrently(
        filepath=input_csv_path, 
        config=config,
        system_prompt_content=system_prompt_content,
        response_field=response_field,
        llm_model_name=llm_model_name,
        api_key_prefix=api_key_prefix,
        tiktoken_encoding_func=tiktoken_encoding_func
    )

    assert processed_df is not None
    # The code is currently including all rows in the output, even from failed chunks
    # So we expect all 100 rows to be present
    assert len(processed_df) == 100

    # Verify that the LLMClient was initialized
    assert MockLLMClient.call_count > 0
    
    # The error message now directly uses the exception message from the mock
    expected_error_message_pattern = "Simulated API failure for chunk large_chunk_part1"

    successful_rows_count = 0
    failed_rows_count = 0

    assert len(ids_in_failed_chunk) > 0, "Mock should have identified some IDs for the failing chunk"
    
    expected_failed_rows_count = len(ids_in_failed_chunk)

    for _, row in processed_df.iterrows():
        original_id = str(row['id'])
        actual_response = str(row[response_field])

        if original_id in ids_in_failed_chunk:
            assert expected_error_message_pattern in actual_response, \
                f"Row {original_id} (failed chunk) expected error '{expected_error_message_pattern}', got '{actual_response}'"
            failed_rows_count += 1
        else:
            assert "Mocked successful response for id" in actual_response, \
                f"Row {original_id} (successful chunk) expected mock success, got '{actual_response}'"
            successful_rows_count += 1
    
    # With 100 total rows and 3 chunks, we expect about 34 rows in the first chunk (which fails)
    # and 33 in each of the other two chunks (which succeed)
    assert failed_rows_count > 0, "Expected some failed rows"
    assert successful_rows_count > 0, "Expected some successful rows"
    assert (failed_rows_count + successful_rows_count) == 100, "Should have 100 rows in total"
    
    output_file_name_template = config["output_options"]["output_filename_template"]
    output_file_name = output_file_name_template.format(original_filename=input_csv_path.stem)
    expected_output_file = Path(temp_test_dir["outputs"]) / output_file_name
    
    if processed_df is not None and not processed_df.empty:
        expected_output_file.parent.mkdir(parents=True, exist_ok=True)
        processed_df.to_csv(expected_output_file, index=False)
        
    assert expected_output_file.exists(), f"Output file {expected_output_file} was not created."
    df_output = pd.read_csv(expected_output_file)
    # We expect all 100 rows in the output, including the ones from the failed chunk
    assert len(df_output) == 100

    output_successful_rows_count = 0
    output_failed_rows_count = 0
    for _, row in df_output.iterrows():
        original_id = str(row['id'])
        actual_response = str(row[response_field])
        if original_id in ids_in_failed_chunk:
            assert expected_error_message_pattern in actual_response
            output_failed_rows_count += 1
        else:
            assert "Mocked successful response for id" in actual_response
            output_successful_rows_count += 1
    
    # With 100 total rows, we expect all rows to be in the output
    # The failed rows should match our expected count
    assert output_failed_rows_count == expected_failed_rows_count, \
        f"Expected {expected_failed_rows_count} failed rows, got {output_failed_rows_count}"
    # The rest should be successful
    assert (output_failed_rows_count + output_successful_rows_count) == 100, \
        f"Expected 100 total rows, got {output_failed_rows_count + output_successful_rows_count}"
    assert output_successful_rows_count == (100 - expected_failed_rows_count), \
        f"Expected {100 - expected_failed_rows_count} successful rows, got {output_successful_rows_count}"
