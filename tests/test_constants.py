"""
Unit tests for the constants module.
"""

import os
from pathlib import Path
import pytest

from src.constants import (PROJECT_ROOT, DEFAULT_LOG_DIR, DEFAULT_ARCHIVE_DIR,
                           LOG_DIR, ARCHIVE_DIR, MAX_BATCH_SIZE, DEFAULT_MODEL,
                           DEFAULT_GLOBAL_TOKEN_LIMIT,
                           DEFAULT_SPLIT_TOKEN_LIMIT, DEFAULT_RESPONSE_FIELD)


def test_project_root():
    """Test that PROJECT_ROOT points to the correct directory"""
    assert PROJECT_ROOT.exists()
    assert PROJECT_ROOT.is_dir()
    # Project root should contain common dirs like 'src', 'tests', etc.
    assert (PROJECT_ROOT / 'src').exists()
    assert (PROJECT_ROOT / 'tests').exists()
    assert (PROJECT_ROOT / 'config').exists() or (PROJECT_ROOT /
                                                  'config').parent.exists()


def test_log_directories():
    """Test that log directory constants are correctly defined"""
    # Test default log directories
    assert DEFAULT_LOG_DIR == PROJECT_ROOT / 'output' / 'logs'
    assert DEFAULT_ARCHIVE_DIR == DEFAULT_LOG_DIR / 'archive'

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
    assert '-' in DEFAULT_MODEL
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
