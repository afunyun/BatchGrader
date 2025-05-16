"""
Unit tests for the log_utils module.
"""

import os
import pytest
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

from src.log_utils import prune_logs_if_needed


@pytest.fixture
def setup_log_dirs(tmp_path):
    """Setup temporary log and archive directories for testing."""
    log_dir = tmp_path / "logs"
    archive_dir = tmp_path / "logs" / "archive"

    # Create directories
    log_dir.mkdir(parents=True)
    archive_dir.mkdir(parents=True)

    # Create test log files with timestamps in filenames
    for i in range(5):
        log_file = log_dir / f"log_202505{i:02d}_120000.log"
        log_file.write_text(f"Log content {i}")

    # Create some archive logs
    for i in range(3):
        archive_file = archive_dir / f"old_log_202504{i:02d}_120000.log"
        archive_file.write_text(f"Old log content {i}")

    return log_dir, archive_dir


def test_prune_logs_if_needed_no_pruning_needed(setup_log_dirs):
    """Test when pruning is not needed because file count is under threshold."""
    log_dir, archive_dir = setup_log_dirs

    # Set max_logs higher than the number of logs we have
    prune_logs_if_needed(str(log_dir),
                         str(archive_dir),
                         max_logs=10,
                         max_archive=10)

    # Should not move any files
    assert len(list(log_dir.glob("*.log"))) == 5
    assert len(list(archive_dir.glob("*.log"))) == 3


def test_prune_logs_if_needed_move_to_archive(setup_log_dirs):
    """Test when logs need to be moved to archive."""
    log_dir, archive_dir = setup_log_dirs

    # Set max_logs lower than the number of logs we have
    prune_logs_if_needed(str(log_dir),
                         str(archive_dir),
                         max_logs=3,
                         max_archive=10)

    # Should move 2 oldest files to archive
    assert len(list(log_dir.glob("*.log"))) == 3

    # Count all files except prune.log
    archive_count = len(
        [f for f in archive_dir.glob("*.log") if f.name != "prune.log"])
    assert archive_count == 5  # 3 original + 2 moved


def test_prune_logs_if_needed_delete_from_archive(setup_log_dirs):
    """Test when archive logs need to be deleted."""
    log_dir, archive_dir = setup_log_dirs

    # Set max_archive lower than the number of archive logs
    prune_logs_if_needed(str(log_dir),
                         str(archive_dir),
                         max_logs=5,
                         max_archive=2)

    # No changes to log_dir, but should delete oldest archive logs
    assert len(list(log_dir.glob("*.log"))) == 5

    # Count all files except prune.log
    archive_count = len(
        [f for f in archive_dir.glob("*.log") if f.name != "prune.log"])
    assert archive_count == 2  # Kept only 2


def test_prune_logs_if_needed_move_and_delete(setup_log_dirs):
    """Test when both moving to archive and deleting from archive are needed."""
    log_dir, archive_dir = setup_log_dirs

    # Set both max_logs and max_archive lower
    prune_logs_if_needed(str(log_dir),
                         str(archive_dir),
                         max_logs=3,
                         max_archive=2)

    # Should move oldest logs and then delete oldest from archive
    assert len(list(log_dir.glob("*.log"))) == 3

    # Count all files except prune.log
    archive_count = len(
        [f for f in archive_dir.glob("*.log") if f.name != "prune.log"])
    assert archive_count == 2  # After moving and pruning


def test_prune_logs_if_needed_creates_directories():
    """Test that directories are created if they don't exist."""
    with patch('os.makedirs') as mock_makedirs, \
         patch('os.listdir', return_value=[]), \
         patch('os.path.isfile', return_value=True):

        prune_logs_if_needed("/nonexistent/logs",
                             "/nonexistent/logs/archive",
                             max_logs=3,
                             max_archive=2)

        # Should have created both directories
        assert mock_makedirs.call_count == 2
        mock_makedirs.assert_any_call("/nonexistent/logs", exist_ok=True)
        mock_makedirs.assert_any_call("/nonexistent/logs/archive",
                                      exist_ok=True)


@patch('os.listdir')
@patch('os.path.isfile')
@patch('os.path.getmtime')
@patch('shutil.move')
@patch('os.remove')
def test_prune_logs_with_mocks(mock_remove, mock_move, mock_getmtime,
                               mock_isfile, mock_listdir):
    """Test pruning logs using mocks to avoid file system operations."""
    # Setup mocks
    mock_listdir.side_effect = lambda path: (
        ['log1.log', 'log2.log', 'log3.log', 'log4.log', '.keep']
        if 'archive' not in path else ['old1.log', 'old2.log', 'old3.log'])
    mock_isfile.return_value = True

    # Return descending modification times (newest first) for main logs
    # and ascending times for archive (oldest first)
    def mock_getmtime_func(path):
        # Normalize path for platform independence
        normalized_path = path.replace('\\', '/')

        if 'archive' not in normalized_path:
            if 'log1.log' in normalized_path: return 1  # oldest
            if 'log2.log' in normalized_path: return 2
            if 'log3.log' in normalized_path: return 3
            if 'log4.log' in normalized_path: return 4  # newest
        else:
            if 'old1.log' in normalized_path: return 1  # oldest
            if 'old2.log' in normalized_path: return 2
            if 'old3.log' in normalized_path: return 3  # newest
        return 0  # for .keep and other files

    mock_getmtime.side_effect = mock_getmtime_func

    # Call the function with max_logs=3 and max_archive=2
    with patch('builtins.open', mock_open()):
        prune_logs_if_needed("/test/logs",
                             "/test/logs/archive",
                             max_logs=3,
                             max_archive=2)

    # Should have moved log1.log to archive (it's the oldest)
    assert mock_move.call_count == 1
    assert '/test/logs/log1.log' in mock_move.call_args[0][0].replace(
        '\\', '/')
    assert '/test/logs/archive/log1.log' in mock_move.call_args[0][1].replace(
        '\\', '/')

    # Should have removed old1.log from archive (it's the oldest)
    assert mock_remove.call_count == 1
    assert '/test/logs/archive/old1.log' in mock_remove.call_args[0][
        0].replace('\\', '/')
