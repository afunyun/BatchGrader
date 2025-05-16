"""
Centralized utilities for LLM client management and token logging.

This module provides functions for LLM client initialization, token usage logging,
and processing with token checking.

Example usage:
    ```python
    # Initialize client
    client = get_llm_client()
    
    # Process with token checking
    result = process_with_token_check(
        df=dataframe,
        system_prompt="You are a helpful assistant",
        response_field="text",
        encoder=encoder,
        token_limit=1000,
        process_func=process_function
    )
    
    if result.success:
        print(f"Processed {len(result.data)} rows")
    ```
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Generic, Optional, TypeVar, Union

import pandas as pd

from llm_client import LLMClient
from logger import logger

# Type variable for DataFrame-like objects
DataFrameT = TypeVar('DataFrameT', bound=pd.DataFrame)
T = TypeVar('T', bound=pd.DataFrame)


@dataclass
class ProcessingResult(Generic[T]):
    """Generic result container for processing operations.
    
    Attributes:
        success: Whether the processing was successful
        data: Processed data (if successful)
        error: Exception that occurred (if any)
        token_usage: Dictionary containing token usage statistics
        start_time: Timestamp when processing started
        end_time: Timestamp when processing ended
        duration: Processing duration in seconds (calculated if end_time is set)
    """
    success: bool
    data: Optional[T] = None
    error: Optional[Exception] = None
    token_usage: Dict[str, Any] = field(default_factory=dict)
    start_time: Optional[float] = field(
        default_factory=lambda: datetime.now(timezone.utc).timestamp())
    end_time: Optional[float] = None
    duration: Optional[float] = None

    def __post_init__(self):
        if self.end_time is not None and self.start_time is not None:
            self.duration = self.end_time - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary.
        
        Returns:
            Dictionary representation of the result
        """
        return {
            'success': self.success,
            'data':
            self.data.to_dict('records') if self.data is not None else None,
            'error': str(self.error) if self.error else None,
            'token_usage': self.token_usage,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration
        }

    @classmethod
    def from_exception(cls, error: Exception) -> 'ProcessingResult':
        """Create a failed result from an exception.
        
        Args:
            error: Exception that occurred
            
        Returns:
            ProcessingResult with error state
        """
        return cls(success=False,
                   error=error,
                   end_time=datetime.now(timezone.utc).timestamp())

    def with_data(self, data: T) -> 'ProcessingResult[T]':
        """Create a new result with updated data.
        
        Args:
            data: New data to include in the result
            
        Returns:
            New ProcessingResult with updated data
        """
        return ProcessingResult(success=self.success,
                                data=data,
                                error=self.error,
                                token_usage=dict(self.token_usage),
                                start_time=self.start_time,
                                end_time=self.end_time,
                                duration=self.duration)


def get_llm_client() -> LLMClient:
    """
    Get a properly initialized LLM client.
    
    Returns:
        Initialized LLMClient instance
    """
    return LLMClient()
