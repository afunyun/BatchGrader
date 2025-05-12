import pytest
import pandas as pd
from pathlib import Path
import shutil
import os

from src.batch_runner import process_file_concurrently
from src.llm_client import LLMClient
from src.config_loader import load_config
from src.logger import logger as global_logger_instance
import tiktoken

@pytest.fixture
def temp_test_dir(tmp_path):
    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir()
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    return {"base": tmp_path, "chunks": chunk_dir, "outputs": output_dir}

@pytest.fixture
def test_config_continue_failure(temp_test_dir):
    config_path = Path("tests/config/test_config_continue_on_failure.yaml")
    cfg = load_config(config_path)

    cfg["input_splitter_options"]["output_base_dir"] = str(temp_test_dir["chunks"])
    cfg["output_options"]["output_base_dir"] = str(temp_test_dir["outputs"])
    if "api_key" not in cfg["llm_client_options"]:
        cfg["llm_client_options"]["api_key"] = "TEST_API_KEY"
    return cfg

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
    input_csv_path = Path("tests/input/chunk_with_failure.csv")
    mocker.patch.object(LLMClient, 'run_batch_job', side_effect=mock_run_batch_job, autospec=True)
    def mock_llm_client_init(self_client, model=None, api_key=None, endpoint=None, logger=None):
        self_client.model = model or test_config_continue_failure['llm_client_options'].get('model')
        self_client.api_key = api_key or test_config_continue_failure['llm_client_options'].get('api_key', "TEST_API_KEY_INIT")
        self_client.endpoint = endpoint
        self_client.logger = logger or global_logger_instance
        self_client.logger.info(f"Mock LLMClient initialized for model {self_client.model}")
    mocker.patch.object(LLMClient, '__init__', side_effect=mock_llm_client_init, autospec=True)
    system_prompt_content = test_config_continue_failure['prompts']['system_prompt']
    response_field = "text"
    llm_model_name = test_config_continue_failure['llm_client_options']['model']
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
        
    LLMClient.__init__.assert_called()
    assert LLMClient.run_batch_job.call_count == 1
    output_file_name = test_config_continue_failure["output_options"]["output_filename_template"].format(original_filename=input_csv_path.stem)
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
