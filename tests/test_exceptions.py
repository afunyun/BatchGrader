"""
Unit tests for the custom exceptions.
"""

import pytest

from batchgrader.exceptions import (  # Add other exceptions from batchgrader.exceptions.py if any are missing
    APIError, BatchGraderError, ChunkingError, DataValidationError,
    FileFormatError, FilePermissionError, FileProcessingError,
    OutputDirectoryError, TokenLimitError, BatchGraderFileNotFoundError,
)

# Attempt to get all exceptions defined in src.exceptions that are subclasses of BatchGraderError
# This is a bit more robust if new exceptions are added.
EXCEPTION_MODULE_CLASSES = [
    BatchGraderError,
    FileProcessingError,
    BatchGraderFileNotFoundError,
    FilePermissionError,
    FileFormatError,
    OutputDirectoryError,
    DataValidationError,
    TokenLimitError,
    ChunkingError,
    APIError,
]

# You might need to check src/exceptions.py for the full list if more exist.
# For now, we'll explicitly list the ones we saw.
# If ConfigError or others exist, please add them to the import and the list below.

ALL_TESTED_EXCEPTIONS = [
    BatchGraderError,
    FileProcessingError,
    BatchGraderFileNotFoundError,
    FilePermissionError,
    FileFormatError,
    OutputDirectoryError,
    DataValidationError,
    TokenLimitError,
    ChunkingError,
    APIError,
    # e.g., ConfigError, if it exists
]


@pytest.mark.parametrize("exception_class", ALL_TESTED_EXCEPTIONS)
def test_exception_can_be_raised_and_caught(exception_class):
    """Test that each custom exception can be raised and caught."""
    error_message = f"This is a test for {exception_class.__name__}"
    with pytest.raises(exception_class, match=error_message):
        raise exception_class(error_message)


def test_file_processing_error_inheritance():
    """Test that FileProcessingError subclasses inherit correctly."""
    assert issubclass(FileProcessingError, BatchGraderError)
    assert issubclass(BatchGraderFileNotFoundError, FileProcessingError)
    assert issubclass(FilePermissionError, FileProcessingError)
    assert issubclass(FileFormatError, FileProcessingError)
    assert issubclass(OutputDirectoryError, FileProcessingError)


def test_other_exceptions_inherit_from_batch_grader_error():
    """Test that other specific exceptions inherit from BatchGraderError."""
    assert issubclass(DataValidationError, BatchGraderError)
    assert issubclass(TokenLimitError, BatchGraderError)
    assert issubclass(ChunkingError, BatchGraderError)
    assert issubclass(APIError, BatchGraderError)
    # Add assertions for other direct BatchGraderError children here
    # e.g., assert issubclass(ConfigError, BatchGraderError)


def test_base_exception_is_generic_exception():
    """Ensure BatchGraderError itself is a generic Exception."""
    assert issubclass(BatchGraderError, Exception)
