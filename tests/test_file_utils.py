"""
Unit tests for the file_utils module.
"""

import os
from pathlib import Path
import pytest
from unittest.mock import patch, mock_open, MagicMock, call, Mock

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
