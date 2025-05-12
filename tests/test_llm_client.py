import pytest
from unittest.mock import MagicMock
from src.llm_client import LLMClient
from src.logger import logger

@pytest.fixture
def llm_client_instance(mocker):
    """
    Provides an LLMClient instance with a mocked OpenAI client.
    """
    mock_openai_class = mocker.patch('openai.OpenAI') 
    mock_openai_instance = MagicMock()
    mock_openai_class.return_value = mock_openai_instance
    
    test_api_key = "fake_test_key"
    test_model = "gpt-3.5-turbo"
    
    client = LLMClient(api_key=test_api_key, model=test_model, logger=logger)
    return client

def test_llm_parse_batch_output_file_success(llm_client_instance, mocker):
    """
    Tests successful parsing of a batch output file.
    """
    client_under_test = llm_client_instance
    
    mock_file_content_text = """{"custom_id": "req1", "response": {"body": {"choices": [{"message": {"content": "Test Response 1"}}]}}}
{"custom_id": "req2", "response": {"body": {"choices": [{"message": {"content": "Test Response 2"}}]}}}"""
    
    mock_content_response = MagicMock()
    mock_content_response.text = mock_file_content_text
    
    mocker.patch.object(client_under_test.client.files, 'content', return_value=mock_content_response)
    
    parsed_results = client_under_test._llm_parse_batch_output_file("fake_output_file_id")
    
    assert "req1" in parsed_results
    assert parsed_results["req1"] == "Test Response 1"
    assert "req2" in parsed_results
    assert parsed_results["req2"] == "Test Response 2"
    client_under_test.client.files.content.assert_called_once_with("fake_output_file_id")

def test_llm_parse_batch_output_file_with_errors(llm_client_instance, mocker):
    """
    Tests parsing of a batch output file that includes errors.
    """
    client_under_test = llm_client_instance
    
    mock_file_content_text = """{"custom_id": "req1", "response": {"body": {"choices": [{"message": {"content": "Test Response 1"}}]}}}
{"custom_id": "req_error", "error": {"message": "This is a test error", "code": "test_err_code"}}
{"custom_id": "req3", "response": {"body": {"choices": [{"message": {"content": "Test Response 3"}}]}}}"""
    
    mock_content_response = MagicMock()
    mock_content_response.text = mock_file_content_text
    mocker.patch.object(client_under_test.client.files, 'content', return_value=mock_content_response)
    
    parsed_results = client_under_test._llm_parse_batch_output_file("fake_output_file_id_with_errors")
    
    assert parsed_results["req1"] == "Test Response 1"
    assert "Error: test_err_code - This is a test error" in parsed_results["req_error"]
    assert parsed_results["req3"] == "Test Response 3"
    client_under_test.client.files.content.assert_called_once_with("fake_output_file_id_with_errors")

def test_llm_parse_batch_output_file_critical_failure(llm_client_instance, mocker):
    """
    Tests behavior when retrieving the batch output file itself fails.
    """
    client_under_test = llm_client_instance
    
    mocker.patch.object(client_under_test.client.files, 'content', side_effect=Exception("API network error"))
    
    with pytest.raises(IOError, match="Failed to retrieve or parse batch output file"):
        client_under_test._llm_parse_batch_output_file("fake_output_file_id_fail")
    
    client_under_test.client.files.content.assert_called_once_with("fake_output_file_id_fail")
