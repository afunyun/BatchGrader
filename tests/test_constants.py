"""
Unit tests for the constants module.
"""

from batchgrader.constants import (
    ARCHIVE_DIR,
    BATCH_API_ENDPOINT,
    DEFAULT_ARCHIVE_DIR,
    DEFAULT_BATCH_DESCRIPTION,
    DEFAULT_EVENT_LOG_PATH,
    DEFAULT_GLOBAL_TOKEN_LIMIT,
    DEFAULT_LOG_DIR,
    DEFAULT_MODEL,
    DEFAULT_POLL_INTERVAL,
    DEFAULT_PRICING_CSV_PATH,
    DEFAULT_PROMPTS_FILE,
    DEFAULT_RESPONSE_FIELD,
    DEFAULT_SPLIT_TOKEN_LIMIT,
    DEFAULT_TOKEN_USAGE_LOG_PATH,
    LOG_DIR,
    MAX_BATCH_SIZE,
    PROJECT_ROOT,
)


def test_log_directories():
    """Test that log directory constants are correctly defined"""
    # Test default log directories
    assert DEFAULT_LOG_DIR == PROJECT_ROOT / "output" / "logs"
    assert DEFAULT_ARCHIVE_DIR == DEFAULT_LOG_DIR / "archive"

    # Test that LOG_DIR and ARCHIVE_DIR are initialized to defaults
    assert LOG_DIR == DEFAULT_LOG_DIR
    assert ARCHIVE_DIR == DEFAULT_ARCHIVE_DIR


def test_batch_size():
    """Test that MAX_BATCH_SIZE is a positive integer"""
    assert isinstance(MAX_BATCH_SIZE, int)
    assert MAX_BATCH_SIZE > 0


def test_default_model():
    """Test that DEFAULT_MODEL is a non-empty string"""
    assert isinstance(DEFAULT_MODEL, str)
    assert DEFAULT_MODEL != ""
    # Should contain version and date in the model name
    assert "-" in DEFAULT_MODEL
    assert any(char.isdigit() for char in DEFAULT_MODEL)


def test_token_limits():
    """Test that token limits are positive integers with expected magnitudes"""
    assert isinstance(DEFAULT_GLOBAL_TOKEN_LIMIT, int)
    assert DEFAULT_GLOBAL_TOKEN_LIMIT > 0
    # Global limit should be larger than split limit
    assert DEFAULT_GLOBAL_TOKEN_LIMIT > DEFAULT_SPLIT_TOKEN_LIMIT

    assert isinstance(DEFAULT_SPLIT_TOKEN_LIMIT, int)
    assert DEFAULT_SPLIT_TOKEN_LIMIT > 0


def test_response_field():
    """Test that DEFAULT_RESPONSE_FIELD is a non-empty string"""
    assert isinstance(DEFAULT_RESPONSE_FIELD, str)
    assert DEFAULT_RESPONSE_FIELD != ""


def test_newly_imported_constants():
    """Test that newly imported constants are defined and have correct types/values."""
    from pathlib import Path

    assert isinstance(DEFAULT_TOKEN_USAGE_LOG_PATH, Path)
    assert isinstance(DEFAULT_EVENT_LOG_PATH, Path)
    assert isinstance(DEFAULT_PRICING_CSV_PATH, Path)

    assert isinstance(BATCH_API_ENDPOINT, str)
    assert BATCH_API_ENDPOINT != ""

    assert isinstance(DEFAULT_PROMPTS_FILE, Path)
    assert isinstance(DEFAULT_BATCH_DESCRIPTION, str)
    assert DEFAULT_BATCH_DESCRIPTION != ""

    assert isinstance(DEFAULT_POLL_INTERVAL, int)
    assert DEFAULT_POLL_INTERVAL > 0
