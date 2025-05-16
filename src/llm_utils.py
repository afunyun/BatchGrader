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
from typing import Dict, Any, Optional, Callable, Tuple, Union, TypeVar, Generic, TypeVar as TypeVarT
import pandas as pd
from datetime import datetime, timezone
from typing import ClassVar, Type

from .llm_client import LLMClient
from .logger import logger
from .token_tracker import update_token_log, log_token_usage_event
from .token_utils import create_token_counter, count_tokens_in_df, calculate_token_stats, check_token_limit, get_token_count_message
from .constants import DEFAULT_MODEL

# Type variable for DataFrame-like objects
DataFrameT = TypeVar('DataFrameT', bound=pd.DataFrame)

T = TypeVarT('T', bound=pd.DataFrame)

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
    start_time: Optional[float] = field(default_factory=lambda: datetime.now(timezone.utc).timestamp())
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
            'data': self.data.to_dict('records') if self.data is not None else None,
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
        return cls(
            success=False,
            error=error,
            end_time=datetime.now(timezone.utc).timestamp()
        )
    
    def with_data(self, data: T) -> 'ProcessingResult[T]':
        """Create a new result with updated data.
        
        Args:
            data: New data to include in the result
            
        Returns:
            New ProcessingResult with updated data
        """
        return ProcessingResult(
            success=self.success,
            data=data,
            error=self.error,
            token_usage=dict(self.token_usage),
            start_time=self.start_time,
            end_time=self.end_time,
            duration=self.duration
        )

def get_llm_client() -> LLMClient:
    """
    Get a properly initialized LLM client.
    
    Returns:
        Initialized LLMClient instance
    """
    return LLMClient()


def log_token_usage(
    api_key: str, 
    model_name: str, 
    input_tokens: int, 
    output_tokens: int = 0, 
    request_id: Optional[str] = None
) -> None:
    """
    Log token usage centrally.
    
    Args:
        api_key: API key used for the request
        model_name: Name of the model used
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        request_id: Optional request ID for tracking
    """
    try:
        # Update running count for token limits
        update_token_log(api_key, input_tokens)
        
        # Log detailed usage event
        log_token_usage_event(
            api_key=api_key,
            model=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            timestamp=None,
            request_id=request_id
        )
        logger.debug("Token usage logged successfully")
    except Exception as e:
        logger.error(f"Failed to log token usage: {e}")


def process_with_token_check(
    df: pd.DataFrame,
    system_prompt: str,
    response_field: str,
    encoder: Any,
    token_limit: int,
    process_func: Callable[[pd.DataFrame], T],
    model_name: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    raise_on_exceed: bool = False,
    request_id: Optional[str] = None
) -> ProcessingResult[T]:
    """
    Process a DataFrame with token checking before sending to API.
    
    Args:
        df: Input DataFrame
        system_prompt: System prompt
        response_field: Field containing response content
        encoder: Tokenizer encoder
        token_limit: Maximum allowed tokens
        process_func: Function to process the DataFrame if token check passes
        model_name: Model name for token logging
        api_key: API key for token logging (if None, get from LLMClient)
        raise_on_exceed: Whether to raise an exception if token limit is exceeded
        request_id: Optional request ID for tracking
        
    Returns:
        ProcessingResult containing the result of the operation
    """
    start_time = datetime.now(timezone.utc).timestamp()
    
    try:
        # Input validation
        if not isinstance(df, pd.DataFrame) or df.empty:
            error_msg = "Input must be a non-empty pandas DataFrame"
            logger.error(error_msg)
            return ProcessingResult[T](success=False, error=ValueError(error_msg))
            
        if response_field not in df.columns:
            error_msg = f"Response field '{response_field}' not found in DataFrame columns"
            logger.error(error_msg)
            return ProcessingResult[T](success=False, error=ValueError(error_msg))
        
        # Calculate token stats
        token_counts = df[response_field].apply(
            lambda x: len(encoder.encode(str(x)))
        )
        
        # Add system prompt tokens (once per request)
        system_tokens = len(encoder.encode(system_prompt))
        total_tokens = token_counts.sum() + system_tokens
        
        # Check token limits
        if total_tokens > token_limit:
            error_msg = (
                f"Token limit exceeded: {total_tokens} tokens "
                f"(limit: {token_limit})"
            )
            logger.warning(error_msg)
            if raise_on_exceed:
                raise ValueError(error_msg)
            return ProcessingResult[T](success=False, error=ValueError(error_msg))
        
        # Log token usage before processing
        log_token_usage(
            api_key=api_key or get_llm_client().api_key,
            model_name=model_name,
            input_tokens=total_tokens,
            request_id=request_id
        )
        
        # Process the data
        processed_data = process_func(df)
        
        # Calculate output tokens (if possible)
        output_tokens = 0
        if hasattr(processed_data, 'columns') and response_field in processed_data.columns:
            output_tokens = processed_data[response_field].apply(
                lambda x: len(encoder.encode(str(x)))
            ).sum()
            
            # Log output token usage
            log_token_usage(
                api_key=api_key or get_llm_client().api_key,
                model_name=model_name,
                input_tokens=0,
                output_tokens=output_tokens,
                request_id=request_id
            )
        
        # Return successful result
        return ProcessingResult[T](
            success=True,
            data=processed_data,
            token_usage={
                'input_tokens': total_tokens,
                'output_tokens': output_tokens,
                'system_tokens': system_tokens,
                'limit': token_limit,
                'limit_exceeded': False
            },
            start_time=start_time,
            end_time=datetime.now(timezone.utc).timestamp()
        )
            
    except Exception as e:
        logger.error(f"Error during token counting: {e}", exc_info=True)
        return ProcessingResult[T](
            success=False,
            error=e,
            token_usage={
                'input_tokens': 0,
                'output_tokens': 0,
                'error': str(e)
            },
            start_time=start_time,
            end_time=datetime.now(timezone.utc).timestamp()
        )
