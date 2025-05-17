"""
Unit tests for the log_utils module.
"""

from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from batchgrader.log_utils import prune_logs_if_needed


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


# @patch('batchgrader.log_utils.Path') # Removing mock-based test due to persistent Path patching issues
# def test_prune_logs_if_needed_creates_directories(mock_path_constructor):
def test_prune_logs_if_needed_creates_directories(tmp_path):
    """Test that directories are created if they don't exist, using real fs."""
    non_existent_log_dir = tmp_path / "nonexistent_logs"
    non_existent_archive_dir = non_existent_log_dir / "archive"

    # Ensure they don't exist initially
    assert not non_existent_log_dir.exists()
    assert not non_existent_archive_dir.exists()

    # Patch open for the prune.log file handling as it's not the focus here
    with patch("builtins.open", mock_open()):
        prune_logs_if_needed(
            str(non_existent_log_dir),
            str(non_existent_archive_dir),
            max_logs=3,
            max_archive=2,
        )

    # Should have created both directories
    assert non_existent_log_dir.exists()
    assert non_existent_log_dir.is_dir()
    assert non_existent_archive_dir.exists()
    assert non_existent_archive_dir.is_dir()

    # Original mock assertions (for reference if patching is fixed later):
    # # Mock Path instances
    # mock_log_path = MagicMock(spec=Path)
    # mock_log_path.exists.return_value = False  # Ensure exists returns False
    # mock_log_path.iterdir.return_value = []  # No files in log dir
    #
    # mock_archive_path = MagicMock(spec=Path)
    # mock_archive_path.exists.return_value = False  # Ensure exists returns False
    # mock_archive_path.iterdir.return_value = []  # No files in archive dir
    #
    # # Set up the Path mock to return our instances
    # # The first call to Path() in prune_logs_if_needed is for log_dir, second for archive_dir
    # mock_path_constructor.side_effect = [mock_log_path, mock_archive_path]
    #
    # # Set up the paths
    # log_dir_str = "/nonexistent/logs"
    # archive_dir_str = "/nonexistent/logs/archive"
    #
    # # Patch open for the prune.log file handling within prune_logs_if_needed
    # with patch('builtins.open', mock_open()):
    #     prune_logs_if_needed(log_dir_str,
    #                          archive_dir_str,
    #                          max_logs=3,
    #                          max_archive=2)
    #
    # # Check that Path was called with the correct arguments
    # expected_path_calls = [call(log_dir_str), call(archive_dir_str)]
    # mock_path_constructor.assert_has_calls(expected_path_calls,
    #                                        any_order=False)
    #
    # # Assert that mkdir was called on the correct mock instances
    # mock_log_path.mkdir.assert_called_once_with(parents=True, exist_ok=True)
    # mock_archive_path.mkdir.assert_called_once_with(parents=True,
    #                                                 exist_ok=True)


@patch("os.listdir")
@patch("os.path.isfile")
@patch("os.path.getmtime")
@patch("shutil.move")
@patch("os.remove")
def test_prune_logs_with_mocks(mock_remove, mock_move, mock_getmtime,
                               mock_isfile, mock_listdir):
    """Test pruning logs using mocks to avoid file system operations."""
    # Setup mocks
    mock_listdir.side_effect = lambda path: ([
        "log1.log", "log2.log", "log3.log", "log4.log", ".keep"
    ] if "archive" not in str(path) else ["old1.log", "old2.log", "old3.log"])
    mock_isfile.return_value = True

    # Return descending modification times (newest first) for main logs
    # and ascending times for archive (oldest first)
    def mock_getmtime_func(path):
        normalized_path = str(path).lower()
        log_times = {
            "current.log": 10,
            "old1.log": 1,
            "old2.log": 2,
            "old3.log": 3,
        }
        # Use dictionary comprehension to find matching log name in the path
        matching_logs = {
            name: time
            for name, time in log_times.items() if name in normalized_path
        }
        # Return the mtime of the first match if any, otherwise default to 0
        return next(iter(matching_logs.values()),
                    0)  # 0 for .keep and other files

    mock_getmtime.side_effect = mock_getmtime_func

    # Call the function with max_logs=3 and max_archive=2
    with patch("builtins.open",
               mock_open()), patch("pathlib.Path") as mock_path:

        # Mock Path instances
        mock_log_path = MagicMock()
        mock_archive_path = MagicMock()
        mock_path.side_effect = [mock_log_path, mock_archive_path]

        # Mock iterdir() to return Path objects
        mock_log_path.iterdir.return_value = [
            Path("/test/logs") / f
            for f in ["log1.log", "log2.log", "log3.log", "log4.log", ".keep"]
        ]
        mock_archive_path.iterdir.return_value = [
            Path("/test/logs/archive") / f
            for f in ["old1.log", "old2.log", "old3.log"]
        ]

        prune_logs_if_needed("/test/logs",
                             "/test/logs/archive",
                             max_logs=3,
                             max_archive=2)
