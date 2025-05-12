import pytest
import pandas as pd
from pathlib import Path
import shutil

from src.batch_runner import process_file_concurrently
from src.llm_client import LLMClient
from src.config_loader import load_config
from src.logger import logger as global_logger_instance # Use 'logger' and alias for minimal changes if preferred, or just use 'logger'
import tiktoken # Added for tiktoken_encoding_func

# Fixture to provide a clean temporary directory for each test
@pytest.fixture
def temp_test_dir(tmp_path):
    # tmp_path is a Path object provided by pytest
    # Create subdirectories for chunks and outputs
    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir()
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    return {"base": tmp_path, "chunks": chunk_dir, "outputs": output_dir}

# Fixture to load the test-specific config and override paths
@pytest.fixture
def test_config_continue_failure(temp_test_dir):
    config_path = Path("tests/config/test_config_continue_on_failure.yaml")
    cfg = load_config(config_path) # Using your existing load_config

    # Override paths to use the temp directory
    cfg["input_splitter_options"]["output_base_dir"] = str(temp_test_dir["chunks"])
    cfg["output_options"]["output_base_dir"] = str(temp_test_dir["outputs"])
    # Ensure API key is set, even if mocked, to pass initial LLMClient setup
    if "api_key" not in cfg["llm_client_options"]:
        cfg["llm_client_options"]["api_key"] = "TEST_API_KEY"
    return cfg


def mock_run_batch_job(self, df, system_prompt_content, response_field_name=None, base_filename_for_tagging=None):
    results_map = {}
    # df is the chunk DataFrame. It should contain a 'custom_id' column added by input_splitter.
    # If not, the test premise or input_splitter's behavior is different from assumption.
    # For now, assume 'custom_id' is present as per earlier test design.
    if 'custom_id' not in df.columns:
        # This would be a problem with test setup or input_splitter assumptions
        self.logger.error("CRITICAL MOCK ERROR: 'custom_id' column missing in DataFrame passed to mock_run_batch_job.")
        # Fallback: try to use 'id' if custom_id is missing, though this might not align with how results are mapped back
        if 'id' in df.columns:
            custom_ids = df["id"].astype(str).tolist() # Using original 'id' as a desperate fallback for custom_id
            df_to_iterate = df.copy()
            df_to_iterate['custom_id'] = custom_ids # Create a temporary custom_id column for the mock's logic
        else:
            raise KeyError("'custom_id' (and fallback 'id') column not found in df for mock_run_batch_job")
    else:
        custom_ids = df["custom_id"].tolist()
        df_to_iterate = df

    # Based on chunk_with_failure.csv (5 rows) and max_rows_per_chunk: 2
    # Expected custom_ids (if generated like 'filename_rowNUMBER'):
    # Chunk 1: chunk_with_failure_row0, chunk_with_failure_row1 (Original IDs: 1, 2)
    # Chunk 2: chunk_with_failure_row2 (Original ID: 3 - This one fails)
    # Chunk 3: chunk_with_failure_row3, chunk_with_failure_row4 (Original IDs: 4, 5)

    # For simplicity, this mock maps original CSV 'id' to failure/success.
    # A more robust mock would inspect df_chunk_with_custom_ids for original content if custom_id is opaque.
    # Let's assume input_splitter.py adds an 'original_id' column to df_chunk_with_custom_ids for easy mapping here.
    # Or, we rely on the custom_id directly if it embeds row index. For this example, we assume custom_id might be like 'file_row_X'
    # and we try to find the one corresponding to original ID '3'.
    
    # A simple way to check which original ID this chunk custom_id corresponds to:
    # Requires `input_splitter.py` to pass through the original ID column or ensure custom_id is predictable.
    # For now, we will assume that `df_chunk_with_custom_ids` contains an 'id' column from the original CSV.
    original_ids_in_chunk = []
    if 'id' in df_to_iterate.columns: # Check in df_to_iterate which has the 'id' column
        original_ids_in_chunk = df_to_iterate['id'].astype(str).tolist()

    simulated_failure_id_str = "3"
    chunk_contains_failure_id = simulated_failure_id_str in original_ids_in_chunk

    if chunk_contains_failure_id:
        self.logger.info(f"Mock: Simulating failure for original_id {simulated_failure_id_str} in chunk {base_filename_for_tagging} with custom_ids: {custom_ids}")
        for idx, custom_id in enumerate(custom_ids):
            # map custom_id back to original_id for this item
            current_original_id = original_ids_in_chunk[idx]
            if current_original_id == simulated_failure_id_str:
                results_map[custom_id] = f"ERROR: Deliberate test failure for ID {simulated_failure_id_str}"
            else: # Should not happen if chunking means ID 3 is alone
                results_map[custom_id] = f"Mocked success for original ID {current_original_id} (in failing chunk's processing)"
    else:
        self.logger.info(f"Mock: Simulating success for chunk {base_filename_for_tagging} with custom_ids: {custom_ids} (original IDs: {original_ids_in_chunk})")
        for idx, custom_id in enumerate(custom_ids):
            current_original_id = original_ids_in_chunk[idx]
            results_map[custom_id] = f"Mocked success for original ID {current_original_id}"
            
    return results_map


def test_continue_on_chunk_failure(mocker, temp_test_dir, test_config_continue_failure):
    input_csv_path = Path("tests/input/chunk_with_failure.csv")

    mocker.patch.object(LLMClient, 'run_batch_job', side_effect=mock_run_batch_job, autospec=True)
    
    def mock_llm_client_init(self_client, model=None, api_key=None, endpoint=None, logger=None):
        self_client.model = model or test_config_continue_failure['llm_client_options'].get('model')
        self_client.api_key = api_key or test_config_continue_failure['llm_client_options'].get('api_key', "TEST_API_KEY_INIT")
        self_client.endpoint = endpoint
        self_client.logger = logger or global_logger_instance # 'logger' here is the param, 'global_logger_instance' is the imported default
        self_client.logger.info(f"Mock LLMClient initialized for model {self_client.model}")

    mocker.patch.object(LLMClient, '__init__', side_effect=mock_llm_client_init, autospec=True)

    # Prepare arguments for process_file_concurrently from the test config
    system_prompt_content = test_config_continue_failure['prompts']['system_prompt']
    response_field = "text"  # From chunk_with_failure.csv's content structure
    llm_model_name = test_config_continue_failure['llm_client_options']['model']
    try:
        tiktoken_encoding_func = tiktoken.encoding_for_model(llm_model_name)
    except KeyError:
        tiktoken_encoding_func = tiktoken.get_encoding("cl100k_base") # Default fallback
    api_key_prefix = "test_key_prefix" # Dummy value

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

    # Expected results map original CSV 'id' to expected result string
    expected_results_by_original_id = {
        "1": "Mocked success for original ID 1",
        "2": "Mocked success for original ID 2",
        "3": f"ERROR: Deliberate test failure for ID 3",
        "4": "Mocked success for original ID 4",
        "5": "Mocked success for original ID 5",
    }

    # Assuming processed_df contains the original 'id' column to verify against
    assert 'id' in processed_df.columns, "Original 'id' column must be in processed_df"
    assert 'result' in processed_df.columns, "'result' column must be in processed_df"

    for index, row in processed_df.iterrows():
        original_id_str = str(row['id'])
        assert row['result'] == expected_results_by_original_id[original_id_str]

    LLMClient.__init__.assert_called()
    assert LLMClient.run_batch_job.call_count == 3 

    output_file_name = test_config_continue_failure["output_options"]["output_filename_template"].format(original_filename=input_csv_path.stem)
    expected_output_file = temp_test_dir["outputs"] / output_file_name
    assert expected_output_file.exists()

    df_output = pd.read_csv(expected_output_file)
    assert len(df_output) == 5
    for index, row in df_output.iterrows():
        original_id_str = str(row['id'])
        assert row['result'] == expected_results_by_original_id[original_id_str]
