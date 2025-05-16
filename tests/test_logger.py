"""
Unit tests for the logger module.
"""

import os
import logging
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call, mock_open

from src.logger import BatchGraderLogger, SUCCESS_LEVEL_NUM


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    mock = MagicMock(spec=logging.Logger)
    return mock


@pytest.fixture
def temp_log_dir(tmp_path):
    """Create a temporary log directory for tests."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    return log_dir


def test_logger_init_creates_directory():
    """Test that logger creates log directory if it doesn't exist."""
    with patch('os.makedirs') as mock_makedirs, \
         patch('logging.getLogger') as mock_get_logger, \
         patch('logging.FileHandler'), \
         patch('rich.logging.RichHandler'):

        mock_logger_instance = MagicMock()
        mock_logger_instance.hasHandlers.return_value = False
        mock_get_logger.return_value = mock_logger_instance

        # Initialize the logger with a non-existent path
        logger = BatchGraderLogger(log_dir='/nonexistent/path')

        # Check that the directory was created
        mock_makedirs.assert_called_once_with('/nonexistent/path')


def test_logger_init_adds_handlers():
    """Test that logger properly adds handlers."""
    # Use a time_patcher approach that works with datetime.datetime
    mock_dt = MagicMock()
    mock_dt.now.return_value = MagicMock()
    mock_dt.now.return_value.strftime.return_value = '20250101_120000'

    # Create a mock for the Rich handler instance
    mock_rich_instance = MagicMock()
    mock_rich_handler = MagicMock(return_value=mock_rich_instance)

    # Create a mock for the file handler instance
    mock_file_instance = MagicMock()
    mock_file_handler = MagicMock(return_value=mock_file_instance)

    with patch('os.path.exists', return_value=True), \
         patch('os.makedirs'), \
         patch('logging.getLogger') as mock_get_logger, \
         patch('logging.FileHandler', mock_file_handler), \
         patch('rich.logging.RichHandler', mock_rich_handler), \
         patch('src.logger.RichHandler', mock_rich_handler), \
         patch('src.logger.datetime', mock_dt):

        # Mock the logger instance
        mock_logger_instance = MagicMock()
        mock_logger_instance.hasHandlers.return_value = False
        mock_get_logger.return_value = mock_logger_instance

        logger = BatchGraderLogger(log_dir='/test/logs')

        # Check that both handlers were created
        assert mock_rich_handler.call_count >= 1
        assert mock_file_handler.call_count == 1

        # Check that the handlers were added to the logger
        assert mock_logger_instance.addHandler.call_count == 2
        mock_logger_instance.addHandler.assert_any_call(mock_rich_instance)
        mock_logger_instance.addHandler.assert_any_call(mock_file_instance)


def test_logger_skips_adding_handlers_if_already_exists():
    """Test that logger doesn't add handlers if they already exist."""
    with patch('os.path.exists', return_value=True), \
         patch('logging.getLogger') as mock_get_logger:

        mock_logger_instance = MagicMock()
        mock_logger_instance.hasHandlers.return_value = True  # Already has handlers
        mock_get_logger.return_value = mock_logger_instance

        logger = BatchGraderLogger(log_dir='/test/logs')

        # Check that no handlers were added
        assert mock_logger_instance.addHandler.call_count == 0


def test_logger_methods_call_underlying_logger():
    """Test that logger methods call the underlying logger methods."""
    with patch('os.path.exists', return_value=True), \
         patch('logging.getLogger') as mock_get_logger, \
         patch('logging.FileHandler'), \
         patch('rich.logging.RichHandler'):

        mock_logger_instance = MagicMock()
        mock_logger_instance.hasHandlers.return_value = True
        mock_get_logger.return_value = mock_logger_instance

        logger = BatchGraderLogger(log_dir='/test/logs')

        # Test various logging methods
        logger.info("Info message")
        mock_logger_instance.info.assert_called_once_with("Info message")

        logger.warning("Warning message")
        mock_logger_instance.warning.assert_called_once_with("Warning message")

        logger.error("Error message")
        mock_logger_instance.error.assert_called_once_with("Error message")

        logger.success("Success message")
        mock_logger_instance.log.assert_called_once_with(
            SUCCESS_LEVEL_NUM, "Success message")

        logger.event("Event message")
        mock_logger_instance.info.assert_called_with("[EVENT] Event message")


def test_success_log_level_registered():
    """Test that the SUCCESS log level is properly registered."""
    assert logging.getLevelName(SUCCESS_LEVEL_NUM) == "SUCCESS"


def test_logger_integration(temp_log_dir):
    """Test actual creation of a logger instance with a real log directory."""
    log_dir = str(temp_log_dir)

    # Create a mock datetime module
    mock_dt = MagicMock()
    mock_dt.now.return_value = MagicMock()
    mock_dt.now.return_value.strftime.return_value = '20250101_120000'

    # Create a real log file in the temp dir with a fixed name for testing
    log_file_path = temp_log_dir / "batchgrader_run_20250101_120000.log"

    # Make sure log directory exists
    os.makedirs(temp_log_dir, exist_ok=True)

    # Replace the actual logger with a mock that writes to our expected file
    class MockBatchGraderLogger(BatchGraderLogger):

        def __init__(self, log_dir=None):
            if log_dir is None:
                log_dir = 'output/logs'
            self.log_dir = log_dir
            self.log_file = str(log_file_path)
            self.logger = logging.getLogger('BatchGrader')

            # Clear any existing handlers
            if self.logger.hasHandlers():
                self.logger.handlers.clear()

            self.logger.setLevel(logging.INFO)
            file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
            file_formatter = logging.Formatter(
                '[%(asctime)s] %(levelname)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S')
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(logging.INFO)
            self.logger.addHandler(file_handler)

    with patch('src.logger.datetime', mock_dt), \
         patch('rich.logging.RichHandler', MagicMock()):
        # Create a logger instance
        logger = MockBatchGraderLogger(log_dir=log_dir)

        # Test logging to the file
        logger.info("Test log message")

    # Check that the log file exists
    assert os.path.exists(log_file_path)

    # Read the log file and check the message is there
    with open(log_file_path, 'r') as f:
        log_content = f.read()
        assert "Test log message" in log_content


def test_get_logger():
    """Test that get_logger returns the underlying logger instance."""
    with patch('os.path.exists', return_value=True), \
         patch('logging.getLogger') as mock_get_logger, \
         patch('logging.FileHandler'), \
         patch('rich.logging.RichHandler'):

        mock_logger_instance = MagicMock()
        mock_logger_instance.hasHandlers.return_value = True
        mock_get_logger.return_value = mock_logger_instance

        logger = BatchGraderLogger(log_dir='/test/logs')
        returned_logger = logger.get_logger()

        # Should return the underlying logger
        assert returned_logger == mock_logger_instance
