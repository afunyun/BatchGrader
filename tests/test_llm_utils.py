"""
Unit tests for the llm_utils module.
"""

from datetime import datetime, timezone
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock, ANY

from src.llm_utils import (
    get_llm_client, 
    log_token_usage, 
    process_with_token_check,
    ProcessingResult
)


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'custom_id': ['1', '2', '3'],
        'response': ['First response', 'Second response', 'Third response'],
        'other_field': ['a', 'b', 'c'],
        'output_tokens': [5, 5, 4]  # Tokens for each response
    })


@pytest.fixture
def mock_encoder():
    """Create a mock encoder for testing."""
    encoder = MagicMock()
    # 1 token per word, plus 1 for the system prompt
    encoder.encode.side_effect = lambda text: [0] * (len(text.split()) + 1)
    return encoder


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client for testing."""
    client = MagicMock()
    client.api_key = "test-api-key"
    return client


def test_processing_result():
    """Test the ProcessingResult dataclass."""
    # Test with minimal required fields
    start_time = datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp()
    result = ProcessingResult[pd.DataFrame](
        success=True,
        data=pd.DataFrame({'a': [1, 2]}),
        token_usage={'input_tokens': 10, 'output_tokens': 5},
        start_time=start_time
    )
    
    assert result.success is True
    assert len(result.data) == 2
    assert result.token_usage['input_tokens'] == 10
    assert result.duration is None  # end_time not set
    
    # Test to_dict method
    result_dict = result.to_dict()
    assert result_dict['success'] is True
    assert len(result_dict['data']) == 2
    assert result_dict['token_usage']['input_tokens'] == 10
    assert result_dict['start_time'] == start_time
    
    # Test from_exception
    exc = ValueError("Test error")
    error_result = ProcessingResult.from_exception(exc)
    assert error_result.success is False
    assert str(error_result.error) == "Test error"
    assert error_result.token_usage == {}
    
    # Test with_data
    new_data = pd.DataFrame({'b': [3, 4, 5]})
    new_result = result.with_data(new_data)
    assert new_result.success is True
    assert len(new_result.data) == 3
    assert result.data is not new_result.data  # Should be a new instance


@patch('src.llm_utils.LLMClient')
def test_get_llm_client(mock_llm_client):
    """Test LLM client retrieval."""
    mock_instance = MagicMock()
    mock_llm_client.return_value = mock_instance
    
    # Test client creation
    client = get_llm_client()
    assert client == mock_instance
    mock_llm_client.assert_called_once()


@patch('src.llm_utils.update_token_log')
@patch('src.llm_utils.log_token_usage_event')
def test_log_token_usage(mock_log_event, mock_update_log):
    """Test token usage logging."""
    # Test with minimal parameters
    log_token_usage("test_key", "test_model", 100)
    mock_update_log.assert_called_with("test_key", 100)
    mock_log_event.assert_called_once()
    args, kwargs = mock_log_event.call_args
    assert kwargs['api_key'] == "test_key"
    assert kwargs['model'] == "test_model"
    assert kwargs['input_tokens'] == 100
    assert kwargs['output_tokens'] == 0
    
    # Test with additional parameters
    mock_update_log.reset_mock()
    mock_log_event.reset_mock()
    log_token_usage("test_key", "test_model", 100, 50, "request123")
    mock_update_log.assert_called_with("test_key", 100)
    mock_log_event.assert_called_once()
    args, kwargs = mock_log_event.call_args
    assert kwargs['api_key'] == "test_key"
    assert kwargs['model'] == "test_model"
    assert kwargs['input_tokens'] == 100
    assert kwargs['output_tokens'] == 50
    assert kwargs['request_id'] == "request123"


@patch('src.llm_utils.log_token_usage')
@patch('src.llm_utils.get_llm_client')
def test_process_with_token_check_success(
    mock_get_client, mock_log_usage, sample_df, mock_encoder
):
    """Test successful processing with token checking."""
    # Setup mocks
    mock_client = MagicMock()
    mock_client.api_key = "test-api-key"
    mock_get_client.return_value = mock_client
    
    # Mock the process function
    mock_process = MagicMock(return_value=sample_df)
    
    # Mock encoder to return 1 token per character for testing
    mock_encoder.encode.side_effect = lambda x: [1] * (len(str(x)) + 1)
    
    # Call the function
    result = process_with_token_check(
        df=sample_df,
        system_prompt="Test prompt",
        response_field="response",
        encoder=mock_encoder,
        token_limit=1000,
        process_func=mock_process,
        model_name="test-model",
        request_id="test-request"
    )
    
    # Verify the result
    assert result.success is True
    assert len(result.data) == 3
    # The mock encoder returns len(str) + 1 tokens for each input
    # System prompt: "Test prompt" -> 11 + 1 = 12 tokens
    # Each response is processed as a string representation of the row
    # Row 1: "custom_id 1 response First response other_field a output_tokens 5"
    # Row 2: "custom_id 2 response Second response other_field b output_tokens 5"
    # Row 3: "custom_id 3 response Third response other_field c output_tokens 4"
    # The actual token count is not critical as it's mocked, but we need to ensure
    # the structure is correct
    assert 'input_tokens' in result.token_usage
    assert 'output_tokens' in result.token_usage
    assert result.token_usage['output_tokens'] >= 14  # At least the sum of output_tokens column
    assert result.error is None
    assert isinstance(result.start_time, float)
    assert isinstance(result.end_time, float)
    assert result.duration > 0
    
    # Verify mocks were called correctly
    mock_process.assert_called_once()
    
    # Verify token logging was called for both input and output
    assert mock_log_usage.call_count == 2


@patch('src.llm_utils.get_llm_client')
def test_process_with_token_check_invalid_input(mock_get_client, mock_encoder, sample_df):
    """Test processing with invalid input."""
    # Setup mock client
    mock_client = MagicMock()
    mock_client.api_key = "test-api-key"
    mock_get_client.return_value = mock_client
    
    # Test with empty DataFrame
    def process_fn(df):
        return df
        
    result = process_with_token_check(
        df=pd.DataFrame(), 
        system_prompt="System prompt", 
        response_field="response", 
        encoder=mock_encoder, 
        token_limit=100,
        process_func=process_fn, 
        model_name="test_model"
    )
    
    assert result.success is False
    assert "must be a non-empty pandas DataFrame" in str(result.error)
    assert result.data is None
    
    # Test with missing response field
    result = process_with_token_check(
        df=sample_df, 
        system_prompt="System prompt", 
        response_field="nonexistent_field", 
        encoder=mock_encoder, 
        token_limit=100,
        process_func=process_fn, 
        model_name="test_model"
    )
    
    assert result.success is False
    assert "not found in DataFrame columns" in str(result.error)
    
    # Test with process function that fails
    def failing_process_fn(df):
        raise ValueError("Process function failed")
    
    # Mock encoder to return 1 token per character
    mock_encoder.encode.side_effect = lambda x: [1] * (len(str(x)) + 1)
    
    result = process_with_token_check(
        df=sample_df, 
        system_prompt="System prompt", 
        response_field="response", 
        encoder=mock_encoder, 
        token_limit=1000,
        process_func=failing_process_fn, 
        model_name="test_model"
    )
    
    assert result.success is False
    assert "Process function failed" in str(result.error)