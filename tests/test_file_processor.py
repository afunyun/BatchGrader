"""
Unit tests for the file_processor module.
"""

import os
import sys
import json
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock, call

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from file_processor import (
    check_token_limits,
    prepare_output_path,
    calculate_and_log_token_usage,
    process_file_common,
    TokenLimitExceeded
)

@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'custom_id': ['1', '2', '3'],
        'response': ['First response', 'Second response', 'Third response'],
        'other_field': ['a', 'b', 'c'],
        'tokens': [10, 15, 8]  # Pre-calculated token counts
    })

@pytest.fixture
def mock_encoder():
    """Create a mock encoder for testing."""
    encoder = MagicMock()
    # 1 token per word, plus 1 for the system prompt
    encoder.encode.side_effect = lambda text: [0] * (len(text.split()) + 1)
    return encoder


@pytest.fixture
def sample_config():
    """Create a sample configuration for testing."""
    return {
        'openai_model_name': 'test-model',
        'force_chunk_count': 0,
        'token_limit': 1000,
        'split_token_limit': 500,
        'llm_output_column_name': 'llm_score',
        'output_dir': 'test_output',
        'system_prompt': 'Test system prompt',
        'response_field': 'response'
    }

@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    return {
        'openai_model_name': 'test-model',
        'force_chunk_count': 0,
        'token_limit': 10000,
        'split_token_limit': 5000,
        'llm_output_column_name': 'llm_score'
    }

def test_check_token_limits(sample_df, mock_encoder):
    """Test token limit checking functionality."""
    # Test when under limit
    is_under_limit, token_stats = check_token_limits(
        sample_df, "System prompt", "response", mock_encoder, 100
    )
    assert is_under_limit is True
    assert 'total' in token_stats
    assert token_stats['total'] > 0  # Should have counted some tokens
    
    # Test when over limit
    is_under_limit, _ = check_token_limits(
        sample_df, "System prompt", "response", mock_encoder, 1
    )
    assert is_under_limit is False
    
    # Test with invalid response field
    with pytest.raises(ValueError, match="not found in DataFrame"):
        check_token_limits(sample_df, "System prompt", "nonexistent", mock_encoder, 100)
    
    # Test with empty DataFrame
    with pytest.raises(ValueError, match="must be a non-empty"):
        check_token_limits(pd.DataFrame(), "System prompt", "response", mock_encoder, 100)

def test_prepare_output_path(mock_config, tmpdir):
    """Test output path preparation."""
    # Test with directory creation
    output_path = prepare_output_path("input/file.txt", tmpdir, "output")
    assert str(output_path).startswith(str(tmpdir))
    assert "output" in str(output_path)
    assert os.path.exists(os.path.dirname(output_path))
    success, result_df = process_file_common(
        "test_file.csv", "output_dir", mock_config,
        "System prompt", "response", mock_encoder, 100
    )
    
    assert success is False
    assert result_df is None 