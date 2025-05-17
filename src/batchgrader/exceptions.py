"""
Custom exceptions for BatchGrader.
"""


class BatchGraderError(Exception):
    """Base exception class for BatchGrader."""

    pass


class FileProcessingError(BatchGraderError):
    """Base class for file processing related errors."""

    pass


class BatchGraderFileNotFoundError(FileProcessingError):
    """Raised when a required file is not found."""

    pass


class FilePermissionError(FileProcessingError):
    """Raised when there are permission issues with file operations."""

    pass


class FileFormatError(FileProcessingError):
    """Raised when there are issues with file format or content."""

    pass


class OutputDirectoryError(FileProcessingError):
    """Raised when there are issues with the output directory."""

    pass


class DataValidationError(BatchGraderError):
    """Raised when data validation fails."""

    pass


class TokenLimitError(BatchGraderError):
    """Raised when token limits are exceeded."""

    pass


class ChunkingError(BatchGraderError):
    """Raised when there are issues with file chunking."""

    pass


class APIError(BatchGraderError):
    """Raised when there are API-related errors."""

    pass


class ConfigurationError(BatchGraderError):
    """Raised when there are configuration-related errors."""

    pass
