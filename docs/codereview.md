# tests/test_batch_job.py

job.status = "error"
job.error_message = "Something went wrong"
job.error_details = "Error details"
assert job.status == "error"
assert job.error_message == "Something went wrong"
assert job.error_details == "Error details"

greptile-apps
commented
2 hours ago
logic: test_batch_job_status_transitions should validate invalid state transitions and verify complete state machine coverage

tests/test_batch_job.py
def \_assert_progress_str_contains(progress_str: str, expected_percent: str, expected_status: str):
"""Helper to assert progress string contains expected values."""
assert expected_percent in progress_str, f"Expected {expected_percent} in progress string"
assert expected_status in progress_str, f"Expected '{expected_status}' in progress string"
return progress_str

greptile-apps
commented
2 hours ago
style: \_assert_progress_str_contains helper returns progress_str but the return value is never used

Suggested change
def \_assert_progress_str_contains(progress_str: str, expected_percent: str, expected_status: str):
"""Helper to assert progress string contains expected values."""
assert expected_percent in progress_str, f"Expected {expected_percent} in progress string"
assert expected_status in progress_str, f"Expected '{expected_status}' in progress string"
return progress_str
def \_assert_progress_str_contains(progress_str: str, expected_percent: str, expected_status: str):
"""Helper to assert progress string contains expected values."""
assert expected_percent in progress_str, f"Expected {expected_percent} in progress string"
assert expected_status in progress_str, f"Expected '{expected_status}' in progress string"
tests/test_batch_runner.py
mock_args_input_file.input_dir = "/nonexistent/dir"
run_batch_processing(mock_args_input_file, basic_config)

## Test output directory creation failure

def mock_mkdir(\*args, \*\*kwargs):
path = kwargs.get('path', None) if kwargs else (args[0] if args else None)
if path and 'output' in str(path):
raise PermissionError("Access denied")

greptile-apps
commented
2 hours ago
style: Path extraction could fail if args is None. Add None check before accessing path

tests/test_check_token_limits.py
def dummy_encoder(prompt, resp, encoder):

## simple count: sum of prompt and response lengths

def counter(row):
return len(prompt) + len(str(row[resp]))
return counter
@ pytest.mark.parametrize("df, system_prompt_content, response_field, encoder, token_limit, exc, msg_substr", [

greptile-apps
commented
2 hours ago
syntax: Extra space between @ and pytest.mark.parametrize should be removed

Suggested change
@ pytest.mark.parametrize("df, system_prompt_content, response_field, encoder, token_limit, exc, msg_substr", [
@pytest.mark.parametrize("df, system_prompt_content, response_field, encoder, token_limit, exc, msg_substr", [
tests/test_check_token_limits.py
assert stats['total'] == float((len("X")+1)+(len("X")+2)+(len("X")+3))
def test_check_token_limits_exceed_limit():
df = pd.DataFrame({'resp':["aaaa"]})

## token count = len(prompt)+len(resp)=1+4=5, limit=4 => exceed

is_valid, stats = check_token_limits(df, "Z", "resp", dummy_encoder, 4, raise_on_error=False)
assert not is_valid

greptile-apps
commented
2 hours ago
logic: raise_on_error=False means errors won't be raised, but the test doesn't verify this behavior explicitly

tests/test_check_token_limits.py
import pandas as pd
from src.file_processor import check_token_limits
def dummy_encoder(prompt, resp, encoder):

## simple count: sum of prompt and response lengths

def counter(row):
return len(prompt) + len(str(row[resp]))
return counter

greptile-apps
commented
2 hours ago
style: dummy_encoder ignores the encoder parameter but includes it in signature - could cause confusion

tests/test_cli_additional.py
("count", "run_count_mode"),
("split", "run_split_mode"),
])
def test_main_modes(monkeypatch, mode, func_name):
monkeypatch.setattr(cli, 'setup_logging', lambda log_dir: None)
monkeypatch.setattr(cli, 'load_config', lambda config_file: {})
calls = {}
monkeypatch.setattr(cli, func_name, lambda args, config: calls.setdefault(func_name, True))

greptile-apps
commented
2 hours ago
style: setdefault() is redundant here since the value is always True - could just use calls[func_name] = True

tests/test_cli.py
str(input_file), '--config',
str(non_existent_config)
]
mock_sys_argv(argv)
with pytest.raises(SystemExit) as e:
cli.main()
assert e.value.code == 0 ## Continues with empty config

greptile-apps
commented
2 hours ago
logic: exit code 0 for config file not found might mask configuration issues - consider using a non-zero exit code when config file explicitly specified but not found

tests/test_constants.py
from src.constants import (PROJECT_ROOT, DEFAULT_LOG_DIR, DEFAULT_ARCHIVE_DIR,
LOG_DIR, ARCHIVE_DIR, MAX_BATCH_SIZE, DEFAULT_MODEL,
DEFAULT_GLOBAL_TOKEN_LIMIT, DEFAULT_SPLIT_TOKEN_LIMIT,
DEFAULT_RESPONSE_FIELD, DEFAULT_TOKEN_USAGE_LOG_PATH,
DEFAULT_EVENT_LOG_PATH, DEFAULT_PRICING_CSV_PATH,
BATCH_API_ENDPOINT, DEFAULT_PROMPTS_FILE,
DEFAULT_BATCH_DESCRIPTION, DEFAULT_POLL_INTERVAL)

greptile-apps
commented
2 hours ago
logic: Several new constants from src.constants are imported but not tested, including DEFAULT_TOKEN_USAGE_LOG_PATH, DEFAULT_EVENT_LOG_PATH, DEFAULT_PRICING_CSV_PATH, BATCH_API_ENDPOINT, DEFAULT_PROMPTS_FILE, DEFAULT_BATCH_DESCRIPTION, and DEFAULT_POLL_INTERVAL

tests/test_constants.py
"""Test that PROJECT_ROOT points to the correct directory"""
assert PROJECT_ROOT.exists()
assert PROJECT_ROOT.is_dir()

## Project root should contain common dirs like 'src', 'tests', etc

assert (PROJECT_ROOT / 'src').exists()
assert (PROJECT_ROOT / 'tests').exists()
assert (PROJECT_ROOT / 'config').exists() or (PROJECT_ROOT /
'config').parent.exists()

greptile-apps
commented
2 hours ago
logic: The config directory check is too lenient - it should not fall back to checking parent directory since config/ should always exist at project root

Suggested change
assert (PROJECT_ROOT / 'config').exists() or (PROJECT_ROOT /
'config').parent.exists()
assert (PROJECT_ROOT / 'config').exists()
tests/test_cost_estimator.py
assert cost == 0.195

## Test with different token counts

cost = CostEstimator.estimate_cost('gpt-3.5-turbo', 2_000_000,
1_000_000)

## Expected: (0.5 _2 + 1.5_ 1) = 2.5

assert cost == 2.5

greptile-apps
commented
2 hours ago
logic: Test only checks exact equality. Should test with floating point tolerance using pytest.approx for price calculations.

Suggested change
def test*estimate_cost_with_valid_model(mock_pricing_csv):
"""Test cost estimation with a valid model."""
with patch('builtins.open', mock_open(read_data=mock_pricing_csv)):
CostEstimator.\_pricing = None ## Reset the class variable ## 1M input tokens, 0.5M output tokens for gpt-4o-2024-08-06
cost = CostEstimator.estimate_cost('gpt-4o-2024-08-06', 1_000_000,
500_000) ## Expected: (1.0 \_1 + 3.0* 0.5) = 2.5
assert cost == 2.5 ## 0.5M input tokens, 0.2M output tokens for gpt-4o-mini-2024-07-18
cost = CostEstimator.estimate*cost('gpt-4o-mini-2024-07-18', 500_000,
200_000) ## Expected: (0.15 \_0.5 + 0.6* 0.2) = 0.195
assert cost == 0.195 ## Test with different token counts
cost = CostEstimator.estimate*cost('gpt-3.5-turbo', 2_000_000,
1_000_000) ## Expected: (0.5 \_2 + 1.5* 1) = 2.5
assert cost == 2.5
def test*estimate_cost_with_valid_model(mock_pricing_csv):
"""Test cost estimation with a valid model."""
with patch('builtins.open', mock_open(read_data=mock_pricing_csv)):
CostEstimator.\_pricing = None ## Reset the class variable ## 1M input tokens, 0.5M output tokens for gpt-4o-2024-08-06
cost = CostEstimator.estimate_cost('gpt-4o-2024-08-06', 1_000_000,
500_000) ## Expected: (1.0 \_1 + 3.0* 0.5) = 2.5
assert cost == pytest.approx(2.5) ## 0.5M input tokens, 0.2M output tokens for gpt-4o-mini-2024-07-18
cost = CostEstimator.estimate*cost('gpt-4o-mini-2024-07-18', 500_000,
200_000) ## Expected: (0.15 \_0.5 + 0.6* 0.2) = 0.195
assert cost == pytest.approx(0.195) ## Test with different token counts
cost = CostEstimator.estimate*cost('gpt-3.5-turbo', 2_000_000,
1_000_000) ## Expected: (0.5 \_2 + 1.5* 1) = 2.5
assert cost == pytest.approx(2.5)
tests/test_cost_estimator.py
def test_csv_path_exists():
"""Test that the CSV path is correctly defined."""
path = CostEstimator.\_csv_path.replace('\\',
'/') ## Normalize path separators
assert path.endswith('docs/pricing.csv')
assert 'docs' in path
assert 'pricing.csv' in path

greptile-apps
commented
2 hours ago
style: Path validation is fragile - only checks string endings. Should use pathlib for robust path comparison.

tests/test_config_loader.py
Test that ensure_config_files creates a default examples.txt if it's missing.
This covers lines 89-92 of config_loader.py.
"""
logger = logging.getLogger("test_config_creation")
caplog.set_level(logging.INFO)
examples_file_path = (PROJECT_ROOT / DEFAULT_CONFIG['examples_dir']).resolve()
examples_file_backup_path = examples_file_path.with_suffix(examples_file_path.suffix + ".test_bak")

greptile-apps
commented
2 hours ago
style: use of str() for path conversion is unnecessary with pathlib - shutil.move accepts Path objects directly

tests/test_config_loader.py
finally:

## Teardown: Remove the examples file that was potentially created by ensure_config_files

if examples_file_path.exists():
os.remove(examples_file_path)

## Restore the backup if it was made

if file_existed_before_test and examples_file_backup_path.exists():
shutil.move(str(examples_file_backup_path), str(examples_file_path))

greptile-apps
commented
2 hours ago
style: same here - str() conversion not needed for shutil.move with Path objects

tests/test_config_loader.py

## Restore the backup if it was made

if file_existed_before_test and examples_file_backup_path.exists():
shutil.move(str(examples_file_backup_path), str(examples_file_path))
elif file_existed_before_test and not examples_file_backup_path.exists():

## This case might occur if the backup move failed or the backup was unexpectedly removed

## For robustness, if the original file existed but backup is gone, try to recreate a default one to leave system in a known state

## Or, simply log a warning that restoration failed

logger.warning(f"Could not restore {examples_file_path} from {examples_file_backup_path} as backup was not found.")

greptile-apps
commented
2 hours ago
style: this warning should include instructions on how to recreate the examples file manually

tests/test_data_loader.py
})
@pytest.fixture
def temp_csv_file(sample_df):
"""Create a temporary CSV file with sample data."""
with tempfile.NamedTemporaryFile(mode="w", suffix=".csv",
delete=False) as tmp:

greptile-apps
commented
2 hours ago
syntax: extra space in indentation of delete=False parameter

tests/test_data_loader.py
save_data(sample_df, "file.txt")
def test_load_nonexistent_file():
"""Test that loading a nonexistent file raises an appropriate error."""
with pytest.raises(
Exception
): ## Could be FileNotFoundError or other exceptions depending on pandas behavior

greptile-apps
commented
2 hours ago
logic: catching generic Exception is too broad - should specifically catch FileNotFoundError since that's the expected exception from pandas

Suggested change
with pytest.raises(
Exception
): ## Could be FileNotFoundError or other exceptions depending on pandas behavior
with pytest.raises(
FileNotFoundError
): ## Expected exception when file doesn't exist
tests/test_evaluator.py
import yaml
from unittest.mock import patch, mock_open
from src.evaluator import load_prompt_template

## The actual default prompt from config_loader

DEFAULT_PROMPT_TEXT = (
'You are an evaluator trying to determins the closeness of a response to a given style, examples of which will follow. Given the following examples, evaluate whether or not the response matches the target style.\n\n'

greptile-apps
commented
2 hours ago
syntax: 'determins' is misspelled in the default prompt text

Suggested change
'You are an evaluator trying to determins the closeness of a response to a given style, examples of which will follow. Given the following examples, evaluate whether or not the response matches the target style.\n\n'
'You are an evaluator trying to determine the closeness of a response to a given style, examples of which will follow. Given the following examples, evaluate whether or not the response matches the target style.\n\n'
tests/test_evaluator.py
def mock_prompts_yaml():
"""Create mock prompts.yaml content."""
return """
batch_evaluation_prompt: |
Please evaluate the following text and provide a score from 1 to 10 based on the provided examples.
{dynamic_examples}
Text: {input}
Score:

greptile-apps
commented
2 hours ago
logic: scoring scale (1-10) differs from both DEFAULT_PROMPT_TEXT (1-5) and batch_evaluation_prompt_generic (1-5), which could cause inconsistent evaluations

tests/test_evaluator.py
with patch('builtins.open', mock_open(read_data="{}")):
with patch('yaml.safe_load', return_value={}):
with patch('config_loader.DEFAULT_PROMPTS', {}):
with patch('sys.stderr'):
with pytest.raises(RuntimeError) as excinfo:
load_prompt_template('missing_prompt')
assert "Failed to load prompt" in str(excinfo.value)
assert "should REALLY never happen" in str(excinfo.value)

greptile-apps
commented
2 hours ago
style: error message 'should REALLY never happen' is unprofessional and should be replaced with a more descriptive message

tests/test_evaluator.py
"""Test handling of YAML parsing errors."""
with patch('pathlib.Path.exists', return_value=True), \
patch('builtins.open', mock_open(read_data="invalid: yaml: content")), \
patch('yaml.safe_load', side_effect=yaml.YAMLError("Invalid YAML")), \
patch('config_loader.DEFAULT_PROMPTS', {'batch_evaluation_prompt': DEFAULT_PROMPT_TEXT}), \
patch('sys.stderr'):
prompt = load_prompt_template('batch_evaluation_prompt')
assert prompt == DEFAULT_PROMPT_TEXT

greptile-apps
commented
2 hours ago
style: test doesn't verify that stderr received the YAML parsing error message

tests/test_exceptions.py
Unit tests for the custom exceptions.
"""
import pytest
from src.exceptions import (
BatchGraderError,
FileProcessingError,
FileNotFoundError, ## Note: This shadows built-in FileNotFoundError if not careful with imports

greptile-apps
commented
2 hours ago
style: Consider renaming custom FileNotFoundError to BatchGraderFileNotFoundError to avoid shadowing built-in exception

tests/test_exceptions.py

## Attempt to get all exceptions defined in src.exceptions that are subclasses of BatchGraderError

## This is a bit more robust if new exceptions are added

EXCEPTION_MODULE_CLASSES = [
BatchGraderError, FileProcessingError, FileNotFoundError,
FilePermissionError, FileFormatError, OutputDirectoryError,
DataValidationError, TokenLimitError, ChunkingError, APIError
]

greptile-apps
commented
2 hours ago
style: EXCEPTION_MODULE_CLASSES is identical to ALL_TESTED_EXCEPTIONS - consider removing one of these redundant lists

Suggested change
EXCEPTION_MODULE_CLASSES = [
BatchGraderError, FileProcessingError, FileNotFoundError,
FilePermissionError, FileFormatError, OutputDirectoryError,
DataValidationError, TokenLimitError, ChunkingError, APIError
]

## Attempt to get all exceptions defined in src.exceptions that are subclasses of BatchGraderError

## This is a bit more robust if new exceptions are added

ALL_TESTED_EXCEPTIONS = [
BatchGraderError, FileProcessingError, FileNotFoundError,
FilePermissionError, FileFormatError, OutputDirectoryError,
DataValidationError, TokenLimitError, ChunkingError, APIError
]
tests/test_file_processor.py

## If not the failing chunk, proceed with normal mock success for all rows in this chunk

self.logger.info(
f"MOCK: Simulating SUCCESS for all rows in chunk '{base_filename_for_tagging}'."
)
processed_rows_list = []

## Unroll loop if possible, or leave as is if mocking DataFrame rows is required for test logic (since this is a mock, the loop is not asserting, just building the output)

result_df = pd.DataFrame(processed_rows_list)

greptile-apps
commented
2 hours ago
logic: Empty processed_rows_list is created but never populated before being converted to DataFrame, which may not properly test the success case

Suggested change
processed_rows_list = [] ## Unroll loop if possible, or leave as is if mocking DataFrame rows is required for test logic (since this is a mock, the loop is not asserting, just building the output).
result_df = pd.DataFrame(processed_rows_list)
processed_rows_list = [{
'custom_id': str(i),
'text': f'Mocked success text for ID {i}',
response_field_name: f'Mocked successful response for ID {i}'
} for i in ids_in_this_chunk] ## Unroll loop if possible, or leave as is if mocking DataFrame rows is required for test logic (since this is a mock, the loop is not asserting, just building the output).
result_df = pd.DataFrame(processed_rows_list)
tests/test_file_processor.py
Args:
mocker: Pytest mocker fixture for mocking dependencies.
sample_df: Fixture providing a sample DataFrame.
Returns:
None
"""
from llm_client import LLMClient

greptile-apps
commented
2 hours ago
style: Import should be from src.llm_client to maintain consistency with other imports

Suggested change
from llm_client import LLMClient
from src.llm_client import LLMClient
tests/test_file_processor.py
print("\nDEBUG INFO - DataFrame columns:", processed_df.columns.tolist())
print("\nDEBUG INFO - DataFrame sample:")
print(processed_df.head())
print("\nResponse field being checked:", response_field)

## Inspect all rows to find examples of success/failure patterns

print("\nDEBUG - All rows with their response values:")

## Debug print loop removed; not needed for assertions or test correctness

greptile-apps
commented
2 hours ago
style: Debug print statements should be removed or converted to proper logging in production code

Suggested change ## Debug: Print the columns and a sample of the data
print("\nDEBUG INFO - DataFrame columns:", processed_df.columns.tolist())
print("\nDEBUG INFO - DataFrame sample:")
print(processed_df.head())
print("\nResponse field being checked:", response_field) ## Inspect all rows to find examples of success/failure patterns
print("\nDEBUG - All rows with their response values:") ## Debug print loop removed; not needed for assertions or test correctness. ## Assertions to verify DataFrame structure and content
assert processed_df.columns.tolist(), "DataFrame should have columns"
assert not processed_df.empty, "DataFrame should not be empty"
assert response_field in processed_df.columns, f"Response field {response_field} should be in columns"
tests/test_file_processor.py
mock_logger.reset_mock()

## Try again with jobs containing valid DataFrames

result = \_pfc_aggregate_and_cleanup(with_data_jobs, "test.csv", "response")

## The function might still return None if there are internal errors

## But we should at least have logs indicating it tried to process the data

assert mock_logger.warning.called or mock_logger.error.called or result is not None

greptile-apps
commented
2 hours ago
style: Check for logger calls is too permissive - should verify specific expected log messages

tests/test_file_utils.py
tests/test_file_utils.py
assert empty_dir.exists()
assert len(list(empty_dir.iterdir())) == 0
def test_prune_nonexistent_dir():
"""Test pruning a non-existent directory."""

## Should not raise an exception

prune_chunked_dir("/path/that/does/not/exist")

greptile-apps
commented
2 hours ago
style: Add assertions to verify no exceptions were raised and function returned gracefully.

tests/test*file_processor_additional.py
output_dir = tmp_path / "out"
os.makedirs(output_dir, exist_ok=True)
result = fp.prepare_output_path("input/testfile.txt", str(output_dir), config)
assert result.endswith("testfile_forced_results.txt")
def test_prepare_output_path_permission_error(monkeypatch):
monkeypatch.setattr(Path, 'mkdir', lambda \*args, \*\*kwargs: (* for \_ in ()).throw(PermissionError("denied")))

greptile-apps
commented
2 hours ago
style: This generator expression is an overly complex way to raise an exception. Consider using raise PermissionError('denied') directly for better readability.

tests/test_file_processor_additional.py

## Tests for calculate_and_log_token_usage

def test_calculate_and_log_token_usage_minimal(monkeypatch, caplog):
monkeypatch.setattr(fp.CostEstimator, 'estimate_cost', lambda model, i, o: 1.23)
calls = []
monkeypatch.setattr(fp, 'log_token_usage_event', lambda \*\*kwargs: calls.append(kwargs))
df = pd.DataFrame({'a': [1, 2]})

greptile-apps
commented
2 hours ago
logic: Test DataFrame has no 'input_tokens' or 'output_tokens' columns which are expected by calculate_and_log_token_usage. Add these columns to properly test token counting.

Suggested change
df = pd.DataFrame({'a': [1, 2]})
df = pd.DataFrame({'a': [1, 2], 'input_tokens': [10, 20], 'output_tokens': [5, 10]})
tests/test_file_processor_additional.py
def test_calculate_and_log_token_usage_logging_error(monkeypatch, caplog):
monkeypatch.setattr(fp.CostEstimator, 'estimate_cost', lambda model, i, o: 0.0)
def_raise(\*args, \*\*kwargs):
raise Exception("log error")
monkeypatch.setattr(fp, 'log_token_usage_event',\_raise)
caplog.set_level('ERROR', logger=fp.logger.name)
df = pd.DataFrame({'a': [1]})

greptile-apps
commented
2 hours ago
logic: Test DataFrame should include expected columns (input_tokens, output_tokens) to properly test error handling path.

Suggested change
df = pd.DataFrame({'a': [1]})
df = pd.DataFrame({'a': [1], 'input_tokens': [10], 'output_tokens': [5]})
tests/test_helpers.py
try:

## If row is a pandas Series with 'text' field

return len(row['text'])
except (TypeError, KeyError):

## Fallback to string length

return len(str(row))
TEST_OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(**file**), 'output'))

greptile-apps
commented
2 hours ago
style: Consider using pathlib.Path instead of os.path for more robust path handling across platforms

tests/test_input_splitter.py
return pd.DataFrame(data)

## Mock token counting function

def mock_count_tokens(row_series):

## Simple mock: count characters in 'text' field, or 10 if 'text' is not present

if 'text' in row_series and pd.notna(row_series['text']):
return len(str(row_series['text']))
return 10 ## Default token count for rows without 'text' or with NaN 'text'

greptile-apps
commented
2 hours ago
logic: mock_count_tokens returns 10 for missing/NaN text fields but doesn't validate if the text field exists in the row_series. Could cause silent failures if text column is missing.

tests/test_llm_utils.py
@pytest.fixture
def mock_llm_client():
"""Create a mock LLM client for testing."""
with patch('src.llm_utils.LLMClient') as mock_client:
mock_instance = MagicMock(spec=LLMClient)
mock_instance.process_batch = AsyncMock(return_value=pd.DataFrame())

greptile-apps
commented
2 hours ago
style: mock_instance.process_batch returns empty DataFrame - should test with actual expected data structure

tests/test_log_utils.py
"""Test when pruning is not needed because file count is under threshold."""
log_dir, archive_dir = setup_log_dirs

## Set max_logs higher than the number of logs we have

prune_logs_if_needed(str(log_dir),
str(archive_dir),
max_logs=10,
max_archive=10)

greptile-apps
commented
2 hours ago
style: inconsistent indentation (5 spaces) used for function arguments

tests/test_log_utils.py

## Normalize path for platform independence

normalized_path = str(path).replace('\\', '/')
if 'archive' not in normalized_path:
if 'log1.log' in normalized_path: return 1 ## oldest
if 'log2.log' in normalized_path: return 2
if 'log3.log' in normalized_path: return 3
if 'log4.log' in normalized_path: return 4 ## newest

greptile-apps
commented
2 hours ago
style: missing test for case where log file name doesn't match any pattern - currently returns 0 silently

tests/test_llm_client.py
assert not result_df['custom_id'].duplicated().any()
assert result_df['custom_id'].iloc[0] == 'test-uuid-123'
def test_prepare_batch_requests_empty_dataframe(llm_client_instance):
"""Test with empty DataFrame."""
df = pd.DataFrame(columns=['response'])

greptile-apps
commented
2 hours ago
logic: Empty test function with no assertions or implementation

Suggested change
def test_prepare_batch_requests_empty_dataframe(llm_client_instance):
"""Test with empty DataFrame."""
df = pd.DataFrame(columns=['response'])
def test_prepare_batch_requests_empty_dataframe(llm_client_instance):
"""Test with empty DataFrame."""
df = pd.DataFrame(columns=['response'])
requests, result_df = llm_client_instance.\_prepare_batch_requests(df, "test prompt", 'response')

    assert len(requests) == 0
    assert len(result_df) == 0
    assert 'custom_id' in result_df.columns

tests/test_llm_client.py

## Call the method

results = llm_client_instance.\_llm_parse_batch_output_file("test_file_id")

## Verify results

assert len(results) == 2
assert results["test1"] == "test content"
assert results["test2"] == "another test"

greptile-apps
commented
2 hours ago
logic: Duplicate test name 'test_llm_parse_batch_output_file_success' - this will cause one test to be skipped

Suggested change
def test_llm_parse_batch_output_file_success(llm_client_instance, mocker):
"""Test successful parsing of batch output file.""" ## Mock the files.content() method to return test data
mock_response = mocker.MagicMock()
mock_response.text = """
{"custom_id": "test1", "response": {"body": {"choices": [{"message": {"content": "test content"}}]}}}
{"custom_id": "test2", "response": {"body": {"choices": [{"message": {"content": "another test"}}]}}}
""".strip()

    mocker.patch.object(llm_client_instance.client.files, 'content', return_value=mock_response)

    ## Call the method
    results = llm_client_instance._llm_parse_batch_output_file("test_file_id")

    ## Verify results
    assert len(results) == 2
    assert results["test1"] == "test content"
    assert results["test2"] == "another test"

def test_llm_parse_batch_output_file_success_with_mocker(llm_client_instance, mocker):
"""Test successful parsing of batch output file.""" ## Mock the files.content() method to return test data
mock_response = mocker.MagicMock()
mock_response.text = """
{"custom_id": "test1", "response": {"body": {"choices": [{"message": {"content": "test content"}}]}}}
{"custom_id": "test2", "response": {"body": {"choices": [{"message": {"content": "another test"}}]}}}
""".strip()

    mocker.patch.object(llm_client_instance.client.files, 'content', return_value=mock_response)

    ## Call the method
    results = llm_client_instance._llm_parse_batch_output_file("test_file_id")

    ## Verify results
    assert len(results) == 2
    assert results["test1"] == "test content"
    assert results["test2"] == "another test"

tests/test_llm_client.py
mock_completed_status = MagicMock()
mock_completed_status.status = "completed"
mock_completed_status.output_file_id = "output_123"
llm_client_instance.client.batches.retrieve.side_effect = [
mock_first_status, mock_completed_status
]

greptile-apps
commented
2 hours ago
style: Test may be flaky - relies on specific order of status updates without verifying intermediate states

tests/test_logger.py
tests/test_logger.py
tests/test_logger.py
tests/test_prompt_utils.py
invalid_yaml.write_text("invalid: yaml: content: [missing bracket")

## Mock the logger to prevent TypeError with level comparison

mock_logger = MagicMock()
with patch('src.prompt_utils.logger', mock_logger), \
pytest.raises(yaml.YAMLError):
load_prompts(invalid_yaml)

greptile-apps
commented
2 hours ago
style: Test should verify the error message content, not just the exception type

tests/test_prompt_utils.py
valid_yaml.write_text(yaml.dump(expected_prompts))

## Mock the logger to prevent TypeError with level comparison

mock_logger = MagicMock()
with patch('src.prompt_utils.logger', mock_logger):
loaded_prompts = load_prompts(valid_yaml)
assert loaded_prompts == expected_prompts

greptile-apps
commented
2 hours ago
style: Should verify that logger.info was called to confirm successful loading was logged

Suggested change ## Mock the logger to prevent TypeError with level comparison
mock_logger = MagicMock()

    with patch('src.prompt_utils.logger', mock_logger):
        loaded_prompts = load_prompts(valid_yaml)
    assert loaded_prompts == expected_prompts
    ## Mock the logger to prevent TypeError with level comparison
    mock_logger = MagicMock()

    with patch('src.prompt_utils.logger', mock_logger):
        loaded_prompts = load_prompts(valid_yaml)
    assert loaded_prompts == expected_prompts
    mock_logger.info.assert_called_once_with(f"Successfully loaded prompts from {valid_yaml}")

tests/test_prompt_utils.py

## Mock the logger to prevent TypeError with level comparison

mock_logger = MagicMock()
with patch('src.prompt_utils.logger', mock_logger), \
patch('builtins.open', side_effect=Exception("Unexpected error")):
with pytest.raises(Exception) as excinfo:
load_prompts(valid_yaml)
assert "Unexpected error" in str(excinfo.value)

greptile-apps
commented
2 hours ago
style: Should verify that logger.error was called with the unexpected error message

Suggested change ## Mock the logger to prevent TypeError with level comparison
mock_logger = MagicMock()
with patch('src.prompt_utils.logger', mock_logger), \
 patch('builtins.open', side_effect=Exception("Unexpected error")):
with pytest.raises(Exception) as excinfo:
load_prompts(valid_yaml)
assert "Unexpected error" in str(excinfo.value) ## Mock the logger to prevent TypeError with level comparison
mock_logger = MagicMock()
with patch('src.prompt_utils.logger', mock_logger), \
 patch('builtins.open', side_effect=Exception("Unexpected error")):
with pytest.raises(Exception) as excinfo:
load_prompts(valid_yaml)
assert "Unexpected error" in str(excinfo.value)
mock_logger.error.assert_called_once_with(f"Error loading prompts from {valid_yaml}: Unexpected error")
tests/test_rich_display.py

## Check that Console.print was called with a Table

mock_print.assert_called_once()
table_arg = mock_print.call_args
assert isinstance(table_arg, Table)
assert table_arg.title == "BatchGrader Job Summary"

## Can't easily check the exact table content without more mocking

greptile-apps
commented
2 hours ago
style: Missing verification of table content - should add assertions for actual table data beyond just checking the type

tests/test_splitter.py
tests/test_splitter.py
tests/test_token_tracker.py
tests/test_token_utils.py
"""Test counting tokens for input including system prompt and user content."""
row = {'text': 'This is user content'}
system_prompt = 'System instruction'

## Should count system tokens (2) + user tokens (5 for the template + 4 for content)

expected_tokens = 2 + 5 + 4
assert count_input_tokens(row, system_prompt, 'text',
mock_encoder) == expected_tokens

greptile-apps
commented
2 hours ago
style: Test assumes fixed template token count (5) but doesn't verify the template being used
