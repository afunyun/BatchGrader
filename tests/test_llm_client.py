import logging
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from batchgrader.exceptions import FileFormatError
from batchgrader.llm_client import LLMClient


@pytest.fixture
def llm_client_instance(tmp_path_factory):
    """
    Provides an LLMClient instance with a mocked OpenAI client.
    """
    with patch("batchgrader.llm_client.OpenAI") as mock_openai_class:
        mock_openai_instance = MagicMock()
        mock_openai_class.return_value = mock_openai_instance

        test_api_key = "fake_test_key"
        yield LLMClient(api_key=test_api_key, model="gpt-3.5-turbo")


def test_run_batch_job_missing_response_column(llm_client_instance):
    """
    Test that run_batch_job raises FileFormatError if the response_field_name column is missing.
    """
    import pandas as pd

    from batchgrader.exceptions import FileFormatError

    # DataFrame without the required response column
    df_missing_col = pd.DataFrame({"input_text": ["test1", "test2"]})
    system_prompt_content = "System prompt"
    missing_field = "nonexistent_response_column"

    with pytest.raises(FileFormatError) as excinfo:
        llm_client_instance.run_batch_job(df_missing_col,
                                          system_prompt_content,
                                          response_field_name=missing_field)
    assert missing_field in str(excinfo.value)
    assert "not found" in str(excinfo.value)


def test_llm_parse_batch_output_file_with_errors(llm_client_instance):
    """
    Tests parsing of a batch output file that includes errors.
    """
    client_under_test = llm_client_instance

    mock_file_content_text = """{"custom_id": "req1", "response": {"body": {"choices": [{"message": {"content": "Test Response 1"}}]}}}
{"custom_id": "req_error", "error": {"message": "This is a test error", "code": "test_err_code"}}
{"custom_id": "req3", "response": {"body": {"choices": [{"message": {"content": "Test Response 3"}}]}}}"""

    mock_content_response = MagicMock()
    mock_content_response.text = mock_file_content_text

    with patch.object(client_under_test.client.files,
                      "content",
                      return_value=mock_content_response):
        assert_llm_parse_batch_output_file_with_errors(client_under_test)


def assert_llm_parse_batch_output_file_with_errors(client_under_test):
    parsed_results = client_under_test._llm_parse_batch_output_file(
        "fake_output_file_id_with_errors")

    assert parsed_results["req1"] == "Test Response 1"
    assert ("Error: test_err_code - This is a test error"
            in parsed_results["req_error"])
    assert parsed_results["req3"] == "Test Response 3"
    client_under_test.client.files.content.assert_called_once_with(
        "fake_output_file_id_with_errors")


def test_llm_parse_batch_output_file_critical_failure(llm_client_instance):
    """
    Tests behavior when retrieving the batch output file itself fails.
    """
    client_under_test = llm_client_instance

    with patch.object(
            client_under_test.client.files,
            "content",
            side_effect=Exception("API network error"),
    ):
        with pytest.raises(
                IOError,
                match="Failed to retrieve or parse batch output file"):
            client_under_test._llm_parse_batch_output_file(
                "fake_output_file_id_fail")

        client_under_test.client.files.content.assert_called_once_with(
            "fake_output_file_id_fail")


@pytest.fixture
def mock_client():
    return LLMClient(api_key="test_key")


def test_llm_client_initialization(llm_client_instance) -> None:
    r"""Test LLMClient initialization without referencing base_url.\n\n    Args:\n        llm_client_instance: Fixture providing an LLMClient instance.\n\n    Returns:\n        None\n"""
    assert (llm_client_instance.api_key == "fake_test_key"
            )  # Only assert existing attributes
    assert llm_client_instance.model == "gpt-3.5-turbo"


def test_llm_client_init(mocker) -> None:
    """Test LLMClient initialization with correct behaviors.

    Args:
        mocker: Pytest mocker fixture for mocking dependencies.

    Returns:
        None
    """

    # Mock get_config_value to return appropriate values based on the key
    def mock_config_getter(config, key, default=None):
        if key == "max_tokens_per_response":
            return 1000  # Return an integer for max_tokens_per_response
        if key == "poll_interval_seconds":
            return 60  # Return an integer for poll_interval_seconds
        return "config_api_key" if key == "openai_api_key" and not config else default

    mock_get_config = mocker.patch("batchgrader.llm_client.get_config_value")
    mock_get_config.side_effect = mock_config_getter

    # Mock OpenAI and encoder
    mocker.patch("batchgrader.llm_client.OpenAI")
    mocker.patch("batchgrader.llm_client.get_encoder")

    # Test explicit initialization
    client = LLMClient(api_key="valid_key", model="gpt-3.5-turbo")
    assert client.api_key == "valid_key"
    assert client.model == "gpt-3.5-turbo"

    # Test with empty API key falling back to config
    empty_key_client = LLMClient(api_key="", model="gpt-3.5-turbo")
    # Should fall back to config value
    assert empty_key_client.api_key == "config_api_key"


def test_prepare_batch_requests_basic(llm_client_instance):
    """Test basic functionality of _prepare_batch_requests."""
    # Setup test data
    df = pd.DataFrame({
        "response": ["test1", "test2"],
        "other_field": ["a", "b"]
    })
    system_prompt = "Test system prompt"
    response_field = "response"

    # Test with empty DataFrame
    empty_df = pd.DataFrame({"response": [], "other_field": []})

    requests, result_df = llm_client_instance._prepare_batch_requests(
        empty_df, "test prompt", "response")

    assert isinstance(requests, list)
    assert isinstance(result_df, pd.DataFrame)
    assert len(requests) == 0
    assert len(result_df) == 0
    assert "custom_id" in result_df.columns

    # Test with non-empty DataFrame
    requests, result_df = llm_client_instance._prepare_batch_requests(
        df, system_prompt, response_field)

    # Assertions
    assert len(requests) == 2
    assert "custom_id" in result_df.columns
    assert len(result_df) == 2


def test_prepare_batch_requests_empty_dataframe(llm_client_instance):
    """Test with empty DataFrame."""
    df = pd.DataFrame(columns=['response'])
    requests, result_df = llm_client_instance._prepare_batch_requests(
        df, "test prompt", 'response')


def test_prepare_batch_requests_custom_ids(llm_client_instance):
    """Test that custom IDs are generated and mapped correctly."""
    df = pd.DataFrame({"response": ["test1"]})

    with patch("uuid.uuid4") as mock_uuid:
        mock_uuid.return_value = "test-uuid-123"
        _, result_df = llm_client_instance._prepare_batch_requests(
            df, "test prompt", "response")

    assert not result_df["custom_id"].duplicated().any()
    assert result_df["custom_id"].iloc[0] == "test-uuid-123"


def test_llm_parse_batch_output_file_success(llm_client_instance, mocker):
    """Test successful parsing of batch output file."""
    # Mock the files.content() method to return test data
    mock_response = mocker.MagicMock()
    mock_response.text = """
    {"custom_id": "test1", "response": {"body": {"choices": [{"message": {"content": "test content"}}]}}}
    {"custom_id": "test2", "response": {"body": {"choices": [{"message": {"content": "another test"}}]}}}
    """.strip()

    mocker.patch.object(llm_client_instance.client.files,
                        "content",
                        return_value=mock_response)

    # Call the method
    results = llm_client_instance._llm_parse_batch_output_file("test_file_id")

    # Verify results
    assert len(results) == 2
    assert results["test1"] == "test content"
    assert results["test2"] == "another test"
    llm_client_instance.client.files.content.assert_called_once_with(
        "test_file_id")


def test_llm_parse_batch_output_file_missing_custom_id(llm_client_instance,
                                                       mocker, caplog):
    """Test parsing with missing custom_id in response."""
    mock_response = mocker.MagicMock()
    mock_response.text = (
        '{"response": {"body": {"choices": [{"message": {"content": "test"}}]}}}'
    )
    mocker.patch.object(llm_client_instance.client.files,
                        "content",
                        return_value=mock_response)

    with caplog.at_level(logging.WARNING):
        results = llm_client_instance._llm_parse_batch_output_file(
            "test_file_id")
        assert "Warning: Found item in output without custom_id" in caplog.text
    assert len(results) == 0  # Shouldn't add entries without custom_id


def test_llm_parse_batch_output_file_with_error(llm_client_instance, mocker,
                                                caplog):
    """Test parsing response with error."""
    mock_response = mocker.MagicMock()
    mock_response.text = """
    {"custom_id": "err1", "error": {"message": "test error", "code": "invalid_request_error"}}
    """.strip()
    mocker.patch.object(llm_client_instance.client.files,
                        "content",
                        return_value=mock_response)

    with caplog.at_level(logging.WARNING):
        results = llm_client_instance._llm_parse_batch_output_file(
            "test_file_id")
        assert "Request err1 failed: invalid_request_error - test error" in caplog.text
    assert results["err1"] == "Error: invalid_request_error - test error"


def test_llm_parse_batch_output_file_malformed(llm_client_instance, mocker,
                                               caplog):
    """Test parsing malformed response."""
    mock_response = mocker.MagicMock()
    mock_response.text = '{"custom_id": "bad1", "response": {"invalid": "data"}}'
    mocker.patch.object(llm_client_instance.client.files,
                        "content",
                        return_value=mock_response)

    with caplog.at_level(logging.WARNING):
        results = llm_client_instance._llm_parse_batch_output_file(
            "test_file_id")
        assert "Error parsing successful response for custom_id bad1" in caplog.text
    assert results["bad1"] == "Error: Malformed response data"


def test_llm_retrieve_batch_error_file_success(llm_client_instance, mocker,
                                               caplog):
    """Test successful retrieval of error file."""
    mock_response = mocker.MagicMock()
    mock_response.text = "test error content"
    mocker.patch.object(llm_client_instance.client.files,
                        "content",
                        return_value=mock_response)

    with caplog.at_level(logging.INFO):
        llm_client_instance._llm_retrieve_batch_error_file("error_file_id")
        assert "Batch Error File Content" in caplog.text


def test_llm_retrieve_batch_error_file_failure(llm_client_instance, mocker,
                                               caplog):
    """Test failure when retrieving error file."""
    mocker.patch.object(llm_client_instance.client.files,
                        "content",
                        side_effect=Exception("API error"))

    with caplog.at_level(logging.WARNING):
        llm_client_instance._llm_retrieve_batch_error_file("error_file_id")
        assert "Error retrieving batch error file" in caplog.text


def test_process_batch_outputs_completed_with_output(llm_client_instance,
                                                     mocker):
    """Test processing batch outputs for completed job with output file."""
    # Setup test data
    mock_batch = mocker.MagicMock()
    mock_batch.status = "completed"
    mock_batch.output_file_id = "output_123"
    mock_batch.error_file_id = "error_123"

    df = pd.DataFrame({
        "custom_id": ["test1", "test2"],
        "response": ["resp1", "resp2"]
    })

    # Mock methods
    mocker.patch.object(
        llm_client_instance,
        "_llm_parse_batch_output_file",
        return_value={
            "test1": "result1",
            "test2": "result2"
        },
    )
    mocker.patch.object(llm_client_instance, "_llm_retrieve_batch_error_file")

    # Call method
    result_df = llm_client_instance._process_batch_outputs(mock_batch, df)

    # Verify results
    assert isinstance(result_df, pd.DataFrame)
    assert len(result_df) == 2
    assert "llm_score" in result_df.columns
    llm_client_instance._llm_parse_batch_output_file.assert_called_once_with(
        "output_123")
    llm_client_instance._llm_retrieve_batch_error_file.assert_called_once_with(
        "error_123")


def test_process_batch_outputs_completed_no_output(llm_client_instance, mocker,
                                                   caplog):
    """Test processing batch outputs for completed job without output file."""
    mock_batch = mocker.MagicMock()
    mock_batch.status = "completed"
    mock_batch.output_file_id = None

    df = pd.DataFrame({"custom_id": ["test1"], "response": ["resp1"]})

    with caplog.at_level(logging.WARNING):
        result_df = llm_client_instance._process_batch_outputs(mock_batch, df)
        assert "Batch completed, but no output file ID was provided" in caplog.text

    assert isinstance(result_df, pd.DataFrame)
    assert len(result_df) == 1
    assert "llm_score" in result_df.columns
    assert (result_df.iloc[0]["llm_score"] ==
            "Error: Batch completed with no output file")


def test_process_batch_outputs_failed(llm_client_instance, mocker, caplog):
    """Test processing outputs for failed batch job."""
    mock_batch = mocker.MagicMock()
    mock_batch.status = "failed"
    mock_batch.error_file_id = "error_123"

    df = pd.DataFrame({"custom_id": ["test1"], "response": ["resp1"]})

    with caplog.at_level(logging.WARNING):
        result_df = llm_client_instance._process_batch_outputs(mock_batch, df)
        assert "Batch job did not complete successfully" in caplog.text

    assert isinstance(result_df, pd.DataFrame)
    assert len(result_df) == 1
    assert "llm_score" in result_df.columns
    assert result_df.iloc[0]["llm_score"].startswith(
        "Error: Batch job status - ")


def test_prepare_batch_requests_missing_response_field(llm_client_instance):
    """Test behavior when response field is missing."""
    df = pd.DataFrame({"other_field": ["a", "b"]})

    with pytest.raises(FileFormatError):
        llm_client_instance._prepare_batch_requests(df, "test prompt",
                                                    "nonexistent_field")


def test_prepare_batch_requests_custom_system_prompt(llm_client_instance):
    """Test with custom system prompt."""
    df = pd.DataFrame({"response": ["test"]})
    custom_prompt = "Custom system prompt for testing"

    requests, _ = llm_client_instance._prepare_batch_requests(
        df, custom_prompt, "response")

    assert requests[0]["body"]["messages"][0]["content"] == custom_prompt


def test_prepare_batch_requests_max_tokens(llm_client_instance):
    """Test that max_tokens is set correctly from config."""
    df = pd.DataFrame({"response": ["test"]})

    assert_max_tokens_in_batch_requests(500, llm_client_instance, df)
    assert_max_tokens_in_batch_requests(1000, llm_client_instance, df)


# TODO Rename this here and in `test_prepare_batch_requests_max_tokens`
def assert_max_tokens_in_batch_requests(max_tokens, llm_client_instance, df):
    # Test with custom max_tokens
    llm_client_instance.max_tokens = max_tokens
    result, _ = llm_client_instance._prepare_batch_requests(
        df, "test prompt", "response")
    assert result[0]["body"]["max_tokens"] == max_tokens

    return result


def test_upload_batch_input_file_success(llm_client_instance, tmp_path):
    """Test successful file upload with mock file creation."""
    # Setup test data
    requests_data = [{"test": "data1"}, {"test": "data2"}]
    base_filename = "test_batch"

    # Mock the files.create method to return a mock file with ID
    mock_file = MagicMock()
    mock_file.id = "file_123"

    # Mock the file content for verification
    file_content = {}

    def mock_file_create(*args, **kwargs):
        nonlocal file_content
        file_obj = kwargs.get("file")
        assert file_obj is not None, \
            "Expected 'file' keyword argument to be a file-like object, got None"
        data = file_obj.read()
        # Support both bytes and str for test robustness
        file_content = data.decode("utf-8") if isinstance(data,
                                                          bytes) else data
        return mock_file

    llm_client_instance.client.files.create.side_effect = mock_file_create

    # Call the method
    file_id = llm_client_instance._upload_batch_input_file(
        requests_data, base_filename)

    # Assertions
    assert file_id == "file_123"
    llm_client_instance.client.files.create.assert_called_once()

    # Verify the file content
    assert '{"test": "data1"}' in file_content
    assert '{"test": "data2"}' in file_content
    assert "purpose" in llm_client_instance.client.files.create.call_args[1]
    assert llm_client_instance.client.files.create.call_args[1][
        "purpose"] == "batch"


def test_manage_batch_job_success(llm_client_instance):
    """Test successful batch job management with polling."""
    # Setup mock batch object
    mock_batch = MagicMock()
    mock_batch.id = "batch_123"

    # Mock retrieve to return different statuses before completing
    mock_first_status = MagicMock()
    mock_first_status.status = "in_progress"

    mock_completed_status = MagicMock()
    mock_completed_status.status = "completed"
    mock_completed_status.output_file_id = "output_123"

    llm_client_instance.client.batches.retrieve.side_effect = [
        mock_first_status,
        mock_completed_status,
    ]

    # Mock time.sleep to avoid actual waiting
    with patch("time.sleep"):
        # Call the method
        result = llm_client_instance._manage_batch_job("input_123",
                                                       "test_batch")

    # Assertions
    assert result == mock_completed_status
    assert llm_client_instance.client.batches.create.called
    assert llm_client_instance.client.batches.retrieve.call_count == 2


def test_process_batch_outputs_success(llm_client_instance):
    """Test successful processing of batch outputs."""
    # Setup test data
    mock_batch = MagicMock()
    mock_batch.status = "completed"
    mock_batch.output_file_id = "output_123"

    df = pd.DataFrame({
        "custom_id": ["id1", "id2"],
        "response": ["resp1", "resp2"]
    })

    # Mock the parse method to return test data
    with patch.object(llm_client_instance,
                      "_llm_parse_batch_output_file") as mock_parse:
        mock_parse.return_value = {"id1": "result1", "id2": "result2"}

        # Call the method
        result = llm_client_instance._process_batch_outputs(mock_batch, df)

        # Assertions
        assert "llm_score" in result.columns
        assert result[result["custom_id"] ==
                      "id1"]["llm_score"].iloc[0] == "result1"
        assert result[result["custom_id"] ==
                      "id2"]["llm_score"].iloc[0] == "result2"


def test_process_batch_outputs_with_errors(llm_client_instance):
    """Test processing batch outputs with various error conditions."""
    # Setup test data with error cases
    mock_batch = MagicMock()
    mock_batch.status = "completed"
    mock_batch.output_file_id = "output_123"

    df = pd.DataFrame({
        "custom_id": ["id1", "id2", "id3"],
        "response": ["resp1", "resp2", "resp3"]
    })

    # Mock the parse method to return partial results with errors
    with patch.object(llm_client_instance,
                      "_llm_parse_batch_output_file") as mock_parse:
        _extracted_from_test_process_batch_outputs_with_errors_16(
            mock_parse, llm_client_instance, mock_batch, df)


def _extracted_from_test_process_batch_outputs_with_errors_16(
        mock_parse, llm_client_instance, mock_batch, df):
    mock_parse.return_value = {
        "id1": "result1",
        "id2": "Error: Test error",
        # id3 is missing to test missing results
    }

    # Call the method
    result = llm_client_instance._process_batch_outputs(mock_batch, df)

    # Assertions
    assert "llm_score" in result.columns
    assert result[result["custom_id"] ==
                  "id1"]["llm_score"].iloc[0] == "result1"
    assert ("Error: Test error"
            in result[result["custom_id"] == "id2"]["llm_score"].iloc[0])
    assert ("Error: Result not found"
            in result[result["custom_id"] == "id3"]["llm_score"].iloc[0])


def test_run_batch_job_integration(llm_client_instance):
    """Test the complete run_batch_job workflow with mocks."""
    # Setup test data
    df = pd.DataFrame({"response": ["test1", "test2"]})
    system_prompt = "Test system prompt"

    # Mock the internal methods
    with patch.object(
            llm_client_instance,
            "_prepare_batch_requests") as mock_prep, patch.object(
                llm_client_instance,
                "_upload_batch_input_file") as mock_upload, patch.object(
                    llm_client_instance,
                    "_manage_batch_job") as mock_manage, patch.object(
                        llm_client_instance,
                        "_process_batch_outputs") as mock_process:

        # Setup mock returns
        mock_prep.return_value = (["req1", "req2"],
                                  df.assign(custom_id=["id1", "id2"]))
        mock_upload.return_value = "file_123"
        mock_manage.return_value = MagicMock()
        mock_process.return_value = df.assign(llm_score=["result1", "result2"])

        # Call the method
        result = llm_client_instance.run_batch_job(df, system_prompt,
                                                   "response", "test_batch")

        # Assertions
        mock_prep.assert_called_once_with(df, system_prompt, "response")
        mock_upload.assert_called_once_with(["req1", "req2"], "test_batch")
        mock_manage.assert_called_once_with("file_123", "test_batch")
        assert "llm_score" in result.columns
        assert len(result) == 2


def test_run_batch_job_default_parameters(llm_client_instance):
    """Test run_batch_job with default parameters."""
    # Setup test data
    df = pd.DataFrame({"response": ["test1"]})
    system_prompt = "Test system prompt"

    # Mock config values
    llm_client_instance.config = {"response_field": "custom_response"}

    with patch.object(
            llm_client_instance,
            "_prepare_batch_requests") as mock_prep, patch.object(
                llm_client_instance,
                "_upload_batch_input_file") as mock_upload, patch.object(
                    llm_client_instance,
                    "_manage_batch_job") as mock_manage, patch(
                        "time.time",
                        return_value=12345), patch("uuid.uuid4",
                                                   return_value="mock-uuid"):

        # Setup mocks
        mock_prep.return_value = (["req1"], df.assign(custom_id=["id1"]))
        mock_upload.return_value = "file_123"
        mock_manage.return_value = MagicMock()

        # Call with minimal parameters
        llm_client_instance.run_batch_job(df, system_prompt)

        # Should use default response field from config
        mock_prep.assert_called_once_with(df, system_prompt, "custom_response")
        # Should generate a default base filename with timestamp
        mock_upload.assert_called_once_with(["req1"], "batch_job_12345")


def test_llm_parse_batch_output_file_error_handling(llm_client_instance,
                                                    caplog):
    """Test error handling in _llm_parse_batch_output_file with invalid JSON."""
    # Mock a file with invalid JSON that will cause a JSONDecodeError
    mock_response = MagicMock()
    mock_response.text = '{"invalid": "json"} invalid line'

    with patch.object(llm_client_instance.client.files,
                      "content",
                      return_value=mock_response):
        # Clear any existing log captures
        caplog.clear()

        # This should raise an IOError due to invalid JSON
        with pytest.raises(
                IOError,
                match="Failed to retrieve or parse batch output file"):
            llm_client_instance._llm_parse_batch_output_file("test_file_id")

        # Verify the error was logged
        assert any("Critical error retrieving or parsing batch output file" in
                   record.message for record in caplog.records)


def test_llm_parse_batch_output_file_valid_data(llm_client_instance):
    """Test successful parsing of valid batch output data."""
    # Mock a file with valid JSON lines
    mock_response = MagicMock()
    mock_response.text = """
    {"custom_id": "test1", "response": {"body": {"choices": [{"message": {"content": "test1"}}]}}}
    {"custom_id": "test2", "error": {"message": "Test error", "code": "test_error"}}
    {"custom_id": "test3", "invalid": "structure"}
    """

    with patch.object(llm_client_instance.client.files,
                      "content",
                      return_value=mock_response):
        result = llm_client_instance._llm_parse_batch_output_file(
            "test_file_id")

        # Check valid response
        assert "test1" in result
        assert result["test1"] == "test1"

        # Check error response
        assert "test2" in result
        assert "Error: test_error - Test error" in result["test2"]

        # Check handling of unknown structure
        assert "test3" in result
        assert "Unknown structure" in result["test3"]


def test_llm_retrieve_batch_error_file_error_handling(llm_client_instance,
                                                      caplog):
    """Test error handling in _llm_retrieve_batch_error_file."""
    # Mock an error when retrieving the error file
    with patch.object(llm_client_instance.client.files,
                      "content",
                      side_effect=Exception("API error")):
        # Clear any existing log captures
        caplog.clear()

        # Should not raise an exception, just log the error
        llm_client_instance._llm_retrieve_batch_error_file("error_123")

        # Verify the error was logged
        assert any("Error retrieving batch error file" in record.message
                   for record in caplog.records)
