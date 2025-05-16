from unittest.mock import MagicMock, patch
import logging
import pytest

from llm_client import LLMClient


@pytest.fixture
def llm_client_instance(tmp_path_factory):
    """
    Provides an LLMClient instance with a mocked OpenAI client.
    """
    with patch('llm_client.OpenAI') as mock_openai_class:
        mock_openai_instance = MagicMock()
        mock_openai_class.return_value = mock_openai_instance

        test_api_key = "fake_test_key"
        test_model = "gpt-3.5-turbo"

        client = LLMClient(api_key=test_api_key, model=test_model)
        yield client


def test_llm_parse_batch_output_file_success(llm_client_instance):
    """
    Tests successful parsing of a batch output file.
    """
    client_under_test = llm_client_instance

    mock_file_content_text = """{"custom_id": "req1", "response": {"body": {"choices": [{"message": {"content": "Test Response 1"}}]}}}
{"custom_id": "req2", "response": {"body": {"choices": [{"message": {"content": "Test Response 2"}}]}}}"""

    mock_content_response = MagicMock()
    mock_content_response.text = mock_file_content_text

    with patch.object(client_under_test.client.files,
                      'content',
                      return_value=mock_content_response):
        parsed_results = client_under_test._llm_parse_batch_output_file(
            "fake_output_file_id")

        assert "req1" in parsed_results
        assert parsed_results["req1"] == "Test Response 1"
        assert "req2" in parsed_results
        assert parsed_results["req2"] == "Test Response 2"
        client_under_test.client.files.content.assert_called_once_with(
            "fake_output_file_id")


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
                      'content',
                      return_value=mock_content_response):
        parsed_results = client_under_test._llm_parse_batch_output_file(
            "fake_output_file_id_with_errors")

        assert parsed_results["req1"] == "Test Response 1"
        assert "Error: test_err_code - This is a test error" in parsed_results[
            "req_error"]
        assert parsed_results["req3"] == "Test Response 3"
        client_under_test.client.files.content.assert_called_once_with(
            "fake_output_file_id_with_errors")


def test_llm_parse_batch_output_file_critical_failure(llm_client_instance):
    """
    Tests behavior when retrieving the batch output file itself fails.
    """
    client_under_test = llm_client_instance

    with patch.object(client_under_test.client.files,
                      'content',
                      side_effect=Exception("API network error")):
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
    """Test LLMClient initialization without referencing base_url.\n\n    Args:\n        llm_client_instance: Fixture providing an LLMClient instance.\n\n    Returns:\n        None\n    """
    assert llm_client_instance.api_key == 'fake_test_key'  # Only assert existing attributes
    assert llm_client_instance.model == 'gpt-3.5-turbo'

def test_llm_client_init(mocker) -> None:
    """Test LLMClient initialization with correct behaviors.
    
    Args:
        mocker: Pytest mocker fixture for mocking dependencies.
    
    Returns:
        None
    """

    # Mock get_config_value to return appropriate values based on the key
    def mock_config_getter(config, key, default=None):
        if key == 'max_tokens_per_response':
            return 1000  # Return an integer for max_tokens_per_response
        if key == 'poll_interval_seconds':
            return 60  # Return an integer for poll_interval_seconds
        if key == 'openai_api_key' and not config:
            return "config_api_key"
        return default

    mock_get_config = mocker.patch('llm_client.get_config_value')
    mock_get_config.side_effect = mock_config_getter

    # Mock OpenAI and encoder
    mocker.patch('llm_client.OpenAI')
    mocker.patch('llm_client.get_encoder')

    # Test explicit initialization
    client = LLMClient(api_key="valid_key", model="gpt-3.5-turbo")
    assert client.api_key == "valid_key"
    assert client.model == "gpt-3.5-turbo"

    # Test with empty API key falling back to config
    empty_key_client = LLMClient(api_key="", model="gpt-3.5-turbo")
    # Should fall back to config value
    assert empty_key_client.api_key == "config_api_key"
