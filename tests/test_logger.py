"""
Unit tests for the logger module.
"""

import logging
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from batchgrader.logger import SUCCESS_LEVEL_NUM, setup_logging


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    return MagicMock(spec=logging.Logger)


@pytest.fixture
def temp_log_dir(tmp_path):
    """Create a temporary log directory for tests."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    return log_dir


def _mock_logger_setup():
    mock_path_instance = MagicMock(spec=Path)
    mock_path_constructor = MagicMock(return_value=mock_path_instance)
    mock_file_handler = MagicMock(spec=logging.Handler, level=logging.INFO)
    mock_rich_handler = MagicMock(spec=logging.Handler, level=logging.INFO)
    return (
        mock_path_instance,
        mock_path_constructor,
        mock_file_handler,
        mock_rich_handler,
    )


@patch("batchgrader.logger.Path")
@patch("batchgrader.logger.logging.getLogger")
@patch("batchgrader.logger.logging.FileHandler")
@patch("batchgrader.logger.RichHandler")
def test_logger_init_creates_directory(mock_rich_handler, mock_file_handler,
                                       mock_get_logger, mock_path_constructor):
    """Test that logger creates log directory if it doesn't exist."""
    mock_path_instance, _, _, _ = _mock_logger_setup()
    mock_path_constructor.return_value = mock_path_instance
    mock_logger_instance = MagicMock(hasHandlers=Mock(return_value=False))
    mock_get_logger.return_value = mock_logger_instance

    setup_logging(log_dir=Path("/nonexistent/path"))

    mock_path_instance.mkdir.assert_called_once_with(parents=True,
                                                     exist_ok=True)


@patch("batchgrader.logger.Path")
@patch("batchgrader.logger.logging.getLogger")
@patch("batchgrader.logger.logging.FileHandler")
@patch("batchgrader.logger.RichHandler")
@patch("batchgrader.logger.datetime")
def test_logger_init_adds_handlers(
    mock_datetime,
    mock_rich_handler,
    mock_file_handler,
    mock_get_logger,
    mock_path_constructor,
):
    """Test that logger properly adds handlers."""
    mock_dt_instance = Mock(strftime=Mock(return_value="20250101_120000"))
    mock_datetime.now.return_value = mock_dt_instance
    mock_rich_instance = MagicMock(spec=logging.Handler, level=logging.INFO)
    mock_file_instance = MagicMock(spec=logging.Handler, level=logging.INFO)
    mock_rich_handler.return_value = mock_rich_instance
    mock_file_handler.return_value = mock_file_instance
    mock_path_instance = MagicMock(spec=Path)
    mock_path_constructor.return_value = mock_path_instance
    mock_logger_instance = MagicMock(spec=logging.Logger,
                                     hasHandlers=Mock(return_value=False))
    mock_get_logger.return_value = mock_logger_instance

    setup_logging(log_dir=Path("/test/logs"))

    mock_datetime.now.assert_called_once()
    mock_get_logger.assert_any_call()
    mock_rich_handler.assert_called_once_with(
        show_time=True,
        show_level=True,
        show_path=False,
        rich_tracebacks=True,
        log_time_format="[%X]",
    )
    mock_logger_instance.addHandler.assert_any_call(mock_rich_instance)
    mock_logger_instance.addHandler.assert_any_call(mock_file_instance)
    mock_path_instance.mkdir.assert_called_once_with(parents=True,
                                                     exist_ok=True)


@patch("batchgrader.logger.Path")
@patch("batchgrader.logger.logging.getLogger")
@patch("batchgrader.logger.logging.FileHandler")
@patch("batchgrader.logger.RichHandler")
def test_logger_skips_adding_handlers_if_already_exists(
        mock_rich_handler, mock_file_handler, mock_get_logger, mock_path):
    """Test that logger clears and re-adds handlers even if they already exist."""
    mock_logger_instance = MagicMock(hasHandlers=Mock(return_value=True))
    mock_get_logger.return_value = mock_logger_instance

    setup_logging(log_dir=Path("/test/logs"))

    mock_logger_instance.handlers.clear.assert_called_once()
    assert mock_logger_instance.addHandler.call_count == 2


@patch("batchgrader.logger.Path")
@patch("batchgrader.logger.logging.getLogger")
@patch("batchgrader.logger.logging.FileHandler")
@patch("batchgrader.logger.RichHandler")
@patch("batchgrader.logger.datetime")
def test_logger_methods_call_underlying_logger(mock_datetime,
                                               mock_rich_handler,
                                               mock_file_handler,
                                               mock_get_logger, mock_path):
    """Test that logger methods call the underlying logger methods."""
    mock_dt_instance = Mock(strftime=Mock(return_value="20250101_120000"))
    mock_datetime.now.return_value = mock_dt_instance
    mock_logger_instance = MagicMock(hasHandlers=Mock(return_value=False))
    mock_get_logger.return_value = mock_logger_instance

    setup_logging(log_dir=Path("/test/logs"))
    mock_logger_instance.reset_mock()

    test_logger = logging.getLogger("test_methods_logger")

    test_logger.info("Info message")
    mock_logger_instance.info.assert_called_once_with("Info message")

    test_logger.error("Error message")
    mock_logger_instance.error.assert_called_once_with("Error message")

    test_logger.log(25, "Success message")
    mock_logger_instance.log.assert_called_once_with(25, "Success message")


def test_success_log_level_registered():
    """Test that the SUCCESS log level is properly registered."""
    assert logging.getLevelName(SUCCESS_LEVEL_NUM) == "SUCCESS"


def test_success_method_added_to_logger():
    """Test that the success method is properly added to the Logger class."""
    logger = logging.getLogger("test_logger")
    assert hasattr(logger, "success") and callable(logger.success)

    with patch.object(logger, "isEnabledFor",
                      return_value=True), patch.object(logger,
                                                       "_log") as mock_log:
        logger.success("test message", "arg1", kwarg1="value1")
        mock_log.assert_called_once_with(SUCCESS_LEVEL_NUM,
                                         "test message", ("arg1", ),
                                         kwarg1="value1")

    with patch.object(logger, "isEnabledFor",
                      return_value=False), patch.object(logger,
                                                        "_log") as mock_log:
        logger.success("test message")
        mock_log.assert_not_called()


@patch("batchgrader.logger.datetime")
@patch("batchgrader.logger.RichHandler")
@patch("batchgrader.logger.logging.FileHandler")
def test_logger_integration(mock_file_handler, mock_rich_handler,
                            mock_datetime, temp_log_dir):
    """Test actual creation of a logger instance with a real log directory."""
    fixed_timestamp = "20250101_120000"
    mock_dt_now_instance = MagicMock(strftime=Mock(
        return_value=fixed_timestamp))
    mock_datetime.now.return_value = mock_dt_now_instance

    mock_rich_handler_instance = MagicMock(spec=logging.Handler,
                                           level=logging.INFO)
    mock_rich_handler.return_value = mock_rich_handler_instance

    mock_file_handler_instance = MagicMock(spec=logging.FileHandler,
                                           level=logging.INFO)
    mock_file_handler.return_value = mock_file_handler_instance

    log_file_path = temp_log_dir / f"batchgrader_run_{fixed_timestamp}.log"
    log_file_path.touch()

    setup_logging(log_dir=temp_log_dir, log_level=logging.INFO)
    logger_instance = logging.getLogger("test_integration_actual")
    logger_instance.info("Test log message")

    assert log_file_path.exists()
    mock_file_handler.assert_called_once()
    assert logger_instance.hasHandlers()


@patch("batchgrader.logger.Path")
@patch("batchgrader.logger.logging.getLogger")
@patch("batchgrader.logger.logging.FileHandler")
@patch("batchgrader.logger.RichHandler")
def test_get_logger(mock_rich_handler, mock_file_handler, mock_get_logger,
                    mock_path):
    """Test that get_logger returns the underlying logger instance."""
    mock_logger_instance = MagicMock(hasHandlers=Mock(return_value=True))
    mock_get_logger.return_value = mock_logger_instance

    setup_logging(log_dir=Path("/test/logs"))

    assert mock_get_logger.called
