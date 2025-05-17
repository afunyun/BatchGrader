"""Unit tests for the logger module."""

import logging
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from batchgrader.logger import SUCCESS_LEVEL_NUM, RichHandler, get_logger, setup_logging


@pytest.fixture
def mock_logger():
    return MagicMock(spec=logging.Logger, hasHandlers=Mock(return_value=False))


@pytest.fixture
def mock_handlers():
    file_handler = MagicMock(spec=logging.Handler, level=logging.INFO)
    rich_handler = MagicMock(spec=RichHandler, level=logging.INFO)
    return file_handler, rich_handler


@pytest.fixture
def patched_logger(mock_logger, mock_handlers):
    file_handler, rich_handler = mock_handlers

    with (
            patch("logging.getLogger", return_value=mock_logger),
            patch("logging.FileHandler",
                  return_value=file_handler) as mock_file_handler,
            patch("batchgrader.logger.RichHandler", return_value=rich_handler)
            as mock_rich_handler,
            patch("batchgrader.logger.Path") as mock_path,
    ):
        mock_path.return_value = MagicMock(spec=Path)
        yield mock_logger, mock_file_handler, mock_rich_handler, mock_path


def test_success_level_registered():
    assert logging.getLevelName(SUCCESS_LEVEL_NUM) == "SUCCESS"


def test_success_method_added():
    logger = logging.getLogger("test_success")
    assert hasattr(logger, "success") and callable(logger.success)

    with (
            patch.object(logger, "isEnabledFor", return_value=True),
            patch.object(logger, "_log") as mock_log,
    ):
        logger.success("test", "arg", kwarg=1)
        mock_log.assert_called_once_with(SUCCESS_LEVEL_NUM,
                                         "test", ("arg", ),
                                         extra=None,
                                         stacklevel=1,
                                         kwarg=1)


def test_setup_logging_creates_directory(patched_logger):
    mock_logger, _, _, mock_path = patched_logger
    mock_path.return_value = MagicMock(spec=Path)

    setup_logging(log_dir=Path("/test/logs"))
    mock_path.return_value.mkdir.assert_called_once_with(parents=True,
                                                         exist_ok=True)


def test_setup_logging_adds_handlers(patched_logger):
    mock_logger, mock_file_handler, mock_rich_handler, _ = patched_logger

    setup_logging(log_dir=Path("/test/logs"))

    mock_rich_handler.assert_called_once_with(
        show_time=True,
        show_level=True,
        show_path=False,
        rich_tracebacks=True,
        log_time_format="[%X]",
    )
    assert mock_logger.addHandler.call_count == 2
    mock_logger.setLevel.assert_called_once()


def test_setup_logging_clears_existing_handlers(patched_logger):
    mock_logger, *_ = patched_logger
    mock_logger.hasHandlers.return_value = True
    mock_logger.handlers = [MagicMock()]

    setup_logging(log_dir=Path("/test/logs"))
    mock_logger.handlers.clear.assert_called_once()


def test_logger_methods_forward_calls(patched_logger):
    mock_logger, *_ = patched_logger
    setup_logging(log_dir=Path("/test/logs"))

    mock_logger.reset_mock()
    logging.getLogger("test").info("test")
    mock_logger.info.assert_called_once_with("test")


def test_logger_integration(patched_logger):
    mock_logger, mock_file_handler, mock_rich_handler, _ = patched_logger

    with patch("batchgrader.logger.datetime") as mock_datetime:
        mock_dt = Mock(strftime=Mock(return_value="20250101_120000"))
        mock_datetime.now.return_value = mock_dt

        setup_logging(log_dir=Path("/test/logs"))

        expected_path = "/test/logs/batchgrader_run_20250101_120000.log"
        mock_file_handler.assert_called_once_with(expected_path,
                                                  encoding="utf-8")
        assert mock_rich_handler.called


def test_get_logger_returns_logger():
    with patch("logging.getLogger") as mock_get:
        logger = get_logger()
        assert logger is not None
        mock_get.assert_called_once()
