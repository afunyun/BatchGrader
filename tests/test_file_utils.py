"""
Unit tests for the file_utils module.
"""

import os
from pathlib import Path
import pytest
from unittest.mock import patch, mock_open, MagicMock

from src.file_utils import prune_chunked_dir


@pytest.fixture
def setup_temp_chunked_dir(tmp_path):
    """Setup a temporary directory structure for testing pruning."""
    chunked_dir = tmp_path / "_chunked"
    chunked_dir.mkdir()

    # Create some test files
    (chunked_dir / "chunk1.csv").write_text("test,data\n1,2\n")
    (chunked_dir / "chunk2.csv").write_text("test,data\n3,4\n")
    (chunked_dir / ".keep").write_text("")  # This file should be preserved

    return chunked_dir


def test_prune_chunked_dir(setup_temp_chunked_dir):
    """Test pruning a chunked directory."""
    chunked_dir = setup_temp_chunked_dir

    # Verify initial state
    assert len(list(chunked_dir.iterdir())) == 3
    assert (chunked_dir / "chunk1.csv").exists()
    assert (chunked_dir / "chunk2.csv").exists()
    assert (chunked_dir / ".keep").exists()

    # Call the prune function
    prune_chunked_dir(str(chunked_dir))

    # Verify that only .keep remains
    assert len(list(chunked_dir.iterdir())) == 1
    assert not (chunked_dir / "chunk1.csv").exists()
    assert not (chunked_dir / "chunk2.csv").exists()
    assert (chunked_dir / ".keep").exists()


def test_prune_empty_chunked_dir(tmp_path):
    """Test pruning an empty chunked directory."""
    empty_dir = tmp_path / "empty_chunked"
    empty_dir.mkdir()

    # Call the prune function on an empty directory
    prune_chunked_dir(str(empty_dir))

    # Verify the directory still exists but is empty
    assert empty_dir.exists()
    assert len(list(empty_dir.iterdir())) == 0


def test_prune_nonexistent_dir():
    """Test pruning a non-existent directory."""
    # Should not raise an exception
    prune_chunked_dir("/path/that/does/not/exist")


@patch('os.path.isdir')
@patch('os.listdir')
@patch('os.path.isfile')
@patch('os.remove')
def test_prune_chunked_dir_with_mocks(mock_remove, mock_isfile, mock_listdir,
                                      mock_isdir):
    """Test pruning a chunked directory using mocks."""
    mock_isdir.return_value = True
    mock_listdir.return_value = [
        'file1.csv', 'file2.csv', '.keep', 'subdirectory'
    ]

    # Only the first two are files, .keep is a file, subdirectory is not
    mock_isfile.side_effect = lambda path: not path.endswith('subdirectory')

    # Use os.path.join to ensure platform-appropriate path separators
    prune_chunked_dir('/mock/chunked/dir')

    # Check that remove was called only for the non-.keep files
    assert mock_remove.call_count == 2

    # Get the actual calls made to mock_remove
    call_args_list = [args[0] for args, _ in mock_remove.call_args_list]

    # Check platform-independently if the right files were removed
    assert any('file1.csv' in arg for arg in call_args_list)
    assert any('file2.csv' in arg for arg in call_args_list)

    # Should not have tried to remove .keep or the subdirectory
    for call in mock_remove.call_args_list:
        args, _ = call
        assert not args[0].endswith('.keep')
        assert not args[0].endswith('subdirectory')
