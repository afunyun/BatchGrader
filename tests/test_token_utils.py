from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from batchgrader.token_utils import (
    TokenLimitExceededError,
    calculate_token_stats,
    check_token_limit,
    count_completion_tokens,
    count_input_tokens,
    count_tokens_in_content,
    count_tokens_in_df,
    create_token_counter,
    get_token_count_message,
)


class MockEncoder:
    """Mock tokenizer that returns predictable token counts for testing."""

    def __init__(self, tokens_per_word=1):
        self.tokens_per_word = tokens_per_word

    def encode(self, text):
        # Simple mock implementation: count words and multiply by tokens_per_word
        return [1] * (len(text.split()) * self.tokens_per_word)


@pytest.fixture
def mock_encoder():
    """Return a mock tokenizer that counts 1 token per word."""
    return MockEncoder(tokens_per_word=1)


@pytest.fixture
def sample_df():
    """Create a sample dataframe for testing."""
    return pd.DataFrame({
        "text": ["This is a test", "Another test with more words", "Short"],
        "llm_score": ["5", "3", "4"],
    })


def test_count_tokens_in_content(mock_encoder):
    """Test counting tokens in a simple content string."""
    # "This is a test" has 4 words, so should return 4 tokens
    assert count_tokens_in_content("This is a test", mock_encoder) == 4

    # Empty string should return 0 tokens
    assert count_tokens_in_content("", mock_encoder) == 0

    # None should be converted to string and have 1 token
    assert count_tokens_in_content(None,
                                   mock_encoder) == 1  # "None" as a string


def test_count_tokens_in_content_error_handling():
    """Test error handling when encoder fails."""
    # Create a mock encoder that raises an exception
    faulty_encoder = MagicMock()
    faulty_encoder.encode.side_effect = Exception("Encoding failed")

    # Mock the logger to prevent TypeError with level comparison
    mock_logger = MagicMock()

    # Should return 0 and log error
    with patch("batchgrader.token_utils.logger", mock_logger):
        assert count_tokens_in_content("Some text", faulty_encoder) == 0

    faulty_encoder.encode.assert_called_once_with("Some text")
    mock_logger.error.assert_called_once()


def test_count_input_tokens(mock_encoder):
    """Test counting tokens for input including system prompt and user content."""
    row = {"text": "This is user content"}
    system_prompt = "System instruction"

    # Should count system tokens (2) + user tokens (5 for the template + 4 for content)
    expected_tokens = 2 + 5 + 4
    assert (count_input_tokens(row, system_prompt, "text",
                               mock_encoder) == expected_tokens)


def test_count_completion_tokens(mock_encoder):
    """Test counting tokens in the LLM completion."""
    row = {"llm_score": "This is a completion"}

    # "This is a completion" has 4 words = 4 tokens
    assert count_completion_tokens(row, mock_encoder) == 4

    # Test with custom field name
    row = {"custom_field": "Custom completion text"}
    assert count_completion_tokens(row, mock_encoder,
                                   field="custom_field") == 3

    # Test with missing field
    row = {"other_field": "Some text"}
    assert count_completion_tokens(row, mock_encoder) == 0

    # Test with None value
    row = {"llm_score": None}
    assert count_completion_tokens(row, mock_encoder) == 0

    # Test with empty string value
    row = {"llm_score": ""}
    assert count_completion_tokens(row, mock_encoder) == 0


def test_create_token_counter(mock_encoder):
    """Test creating a token counter function."""
    system_prompt = "System instruction"
    response_field = "text"

    token_counter = create_token_counter(system_prompt, response_field,
                                         mock_encoder)

    row = {"text": "This is user content"}
    # "System instruction\nThis is user content" has 6 words = 6 tokens
    assert token_counter(row) == 6

    # Test with custom template
    template = "{system} - {user}"
    custom_counter = create_token_counter(system_prompt,
                                          response_field,
                                          mock_encoder,
                                          prompt_template=template)
    # "System instruction - This is user content" has 7 words = 7 tokens (including the "-")
    # trunk-ignore(bandit/B101)
    assert custom_counter(row) == 7


def test_count_tokens_in_df(sample_df, mock_encoder):
    """Test counting tokens for each row in a DataFrame."""
    system_prompt = "System"
    counts = count_tokens_in_df(sample_df, system_prompt, "text", mock_encoder)

    # Should return a Series with token counts
    assert isinstance(counts, pd.Series)
    assert len(counts) == 3

    # First row: "System\nThis is a test" = 5 tokens
    assert counts.iloc[0] == 5

    # Second row: "System\nAnother test with more words" = 6 tokens
    assert counts.iloc[1] == 6

    # Third row: "System\nShort" = 2 tokens
    assert counts.iloc[2] == 2


def test_calculate_token_stats():
    """Test calculating token statistics from a Series."""
    token_counts = pd.Series([10, 20, 30, 40])
    stats = calculate_token_stats(token_counts)

    assert stats["total"] == 100.0
    assert stats["average"] == 25.0
    assert stats["max"] == 40.0


def test_check_token_limit():
    """Test checking if tokens exceed a limit."""
    # Under limit
    stats = {"total": 900, "average": 90, "max": 150}
    assert check_token_limit(stats, 1000) is True

    # Over limit
    stats = {"total": 1100, "average": 110, "max": 200}
    assert check_token_limit(stats, 1000) is False

    # Exactly at limit
    stats = {"total": 1000, "average": 100, "max": 200}
    assert check_token_limit(stats, 1000) is True


def test_check_token_limit_with_exception():
    """Test that check_token_limit raises exception when configured to do so."""
    # Over limit with raise_on_exceed=True
    stats = {"total": 1100, "average": 110, "max": 200}
    with pytest.raises(TokenLimitExceededError) as exc_info:
        check_token_limit(stats, 1000, raise_on_exceed=True)

    assert str(exc_info.value).startswith("Token limit exceeded")


def test_get_token_count_message():
    """Test generating a formatted message with token statistics."""
    stats = {"total": 1234.56, "average": 123.45, "max": 567.89}
    message = get_token_count_message(stats)

    assert "[TOKEN COUNT]" in message
    assert "Total: 1234" in message  # Integer part only
    assert "Avg: 123.5" in message  # Rounded to 1 decimal
    assert "Max: 567" in message  # Integer part only