"""
Unified file processing module to reduce duplication in batch processing logic.

This module provides common abstraction for processing files, both sequentially and concurrently,
centralizing token checking, file handling, and result aggregation functionality.

Example usage:
    ```python
    # Check token limits
    is_valid, stats = check_token_limits(
        df=dataframe,
        system_prompt_content="Your prompt",
        response_field="text",
        encoder=encoder,
        token_limit=1000
    )
    
    # Prepare output path
    output_path = prepare_output_path(
        filepath="input.csv",
        output_dir="output",
        config={"output_format": "csv"}
    )
    ```
"""

import os
import sys
import datetime
import logging
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Union, Callable, Tuple, TypeVar, Generic, List

import pandas as pd

from .batch_job import BatchJob
from .llm_client import LLMClient
from .logger import logger
from .file_utils import prune_chunked_dir
from .data_loader import load_data, save_data
from .input_splitter import split_file_by_token_limit
from .cost_estimator import CostEstimator
from .token_utils import count_input_tokens, count_completion_tokens, create_token_counter
from .token_tracker import update_token_log, log_token_usage_event
from .constants import LOG_DIR, ARCHIVE_DIR, MAX_BATCH_SIZE, DEFAULT_MODEL

# Type variable for DataFrame-like objects
DataFrameT = TypeVar('DataFrameT', bound=pd.DataFrame)

@dataclass
class ProcessingStats:
    """Container for file processing statistics."""
    input_path: str
    output_path: str
    rows_processed: int
    token_usage: Dict[str, int]
    start_time: datetime.datetime
    end_time: Optional[datetime.datetime] = None
    error: Optional[Exception] = None
    
    @property
    def duration(self) -> Optional[float]:
        """Return processing duration in seconds."""
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary."""
        return {
            'input_path': self.input_path,
            'output_path': str(self.output_path) if self.output_path else None,
            'rows_processed': self.rows_processed,
            'token_usage': self.token_usage,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration': self.duration,
            'error': str(self.error) if self.error else None
        }


def check_token_limits(
    df: pd.DataFrame, 
    system_prompt_content: str, 
    response_field: str, 
    encoder: Any, 
    token_limit: int,
    raise_on_error: bool = False
) -> Tuple[bool, Dict[str, float]]:
    """
    Check if token counts in the dataframe exceed the specified limit.
    
    Args:
        df: Input dataframe to check. Must be a non-empty pandas DataFrame.
        system_prompt_content: System prompt content or field name. Must be a non-empty string.
        response_field: Field containing response content. Must be a column in the DataFrame.
        encoder: Tokenizer encoder instance.
        token_limit: Maximum allowed tokens. Must be a positive integer.
        raise_on_error: If True, raises ValueError on validation errors.
        
    Returns:
        Tuple containing:
            - Boolean indicating if tokens are under limit
            - Dictionary with token statistics (total, average, max)
            
    Raises:
        ValueError: If input validation fails and raise_on_error is True.
        TypeError: If input types are incorrect and raise_on_error is True.
    """
    # Input validation
    if not isinstance(df, pd.DataFrame):
        error_msg = f"df must be a pandas DataFrame, got {type(df).__name__}"
        if raise_on_error:
            raise TypeError(error_msg)
        logger.error(error_msg)
        return False, {}
        
    if df.empty:
        error_msg = "DataFrame cannot be empty"
        if raise_on_error:
            raise ValueError(error_msg)
        logger.error(error_msg)
        return False, {}
        
    if not isinstance(system_prompt_content, str) or not system_prompt_content.strip():
        error_msg = "system_prompt_content must be a non-empty string"
        if raise_on_error:
            raise ValueError(error_msg)
        logger.error(error_msg)
        return False, {}
        
    if response_field not in df.columns:
        error_msg = f"response_field '{response_field}' not found in DataFrame columns"
        if raise_on_error:
            raise ValueError(error_msg)
        logger.error(error_msg)
        return False, {}
        
    if not isinstance(token_limit, int) or token_limit <= 0:
        error_msg = f"token_limit must be a positive integer, got {token_limit}"
        if raise_on_error:
            raise ValueError(error_msg)
        logger.error(error_msg)
        return False, {}
    
    try:
        token_counter = create_token_counter(system_prompt_content, response_field, encoder)
        token_counts = df.apply(token_counter, axis=1)
        
        if token_counts.empty:
            return True, {'total': 0, 'average': 0, 'max': 0}
            
        token_stats = {
            'total': float(token_counts.sum()),
            'average': float(token_counts.mean()),
            'max': float(token_counts.max())
        }
        
        is_under_limit = token_stats['total'] <= token_limit
        
        logger.info(f"[TOKEN COUNT] Total: {int(token_stats['total'])}, "
                   f"Avg: {token_stats['average']:.1f}, "
                   f"Max: {int(token_stats['max'])}")
        
        if not is_under_limit:
            error_msg = (
                f"Token limit exceeded: {int(token_stats['total'])} > {token_limit}. "
                "Please reduce your batch size or check your usage."
            )
            logger.error(f"[ERROR] {error_msg}")
            if raise_on_error:
                raise ValueError(error_msg)
        
        return is_under_limit, token_stats
        
    except Exception as e:
        error_msg = f"Error checking token limits: {str(e)}"
        logger.error(error_msg, exc_info=True)
        if raise_on_error:
            raise RuntimeError(error_msg) from e
        return False, {}


def prepare_output_path(
    filepath: str, 
    output_dir: str, 
    config: Dict[str, Any]
) -> str:
    """
    Prepare the output path for saving results.
    
    Args:
        filepath: Original input file path
        output_dir: Directory for output files
        config: Configuration dictionary
        
    Returns:
        Output file path for saving results
    """
    filename = os.path.basename(filepath)
    file_root, file_ext = os.path.splitext(filename)
    
    if 'tests' in filepath.replace('\\', '/').lower():
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        output_dir = os.path.join(project_root, 'tests', 'output')
    
    os.makedirs(output_dir, exist_ok=True)
    
    config_force_chunk = config.get('force_chunk_count', 0)
    if 'legacy' in file_root.lower():
        out_suffix = '_results'
    elif config_force_chunk and config_force_chunk > 1:
        out_suffix = '_forced_results'
    else:
        out_suffix = '_results'
        
    output_filename = f"{file_root}{out_suffix}{file_ext}"
    output_path = os.path.join(output_dir, output_filename)
    
    if os.path.exists(output_path):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{file_root}{out_suffix}_{timestamp}{file_ext}"
        output_path = os.path.join(output_dir, output_filename)
        logger.info(f"Output file already exists. Using new filename: {output_filename}")
        
    return output_path


def calculate_and_log_token_usage(
    df_with_results: pd.DataFrame, 
    system_prompt_content: str, 
    response_field: str, 
    encoder: Any, 
    model_name: str, 
    api_key: str
) -> None:
    """
    Calculate token usage stats and log to the token tracking system.
    
    Args:
        df_with_results: DataFrame with results
        system_prompt_content: System prompt content
        response_field: Response field name
        encoder: Tokenizer encoder
        model_name: Name of the LLM model used
        api_key: API key for token logging
    """
    try:
        input_col_name = 'input_tokens'
        output_col_name = 'output_tokens'
        
        if encoder is not None and (input_col_name not in df_with_results.columns or 
                                  output_col_name not in df_with_results.columns):
            df_with_results[input_col_name] = df_with_results.apply(
                lambda row: count_input_tokens(row, system_prompt_content, response_field, encoder), 
                axis=1
            )
            df_with_results[output_col_name] = df_with_results.apply(
                lambda row: count_completion_tokens(row, encoder), 
                axis=1
            )
            
        n_input_tokens = int(df_with_results[input_col_name].sum()) if input_col_name in df_with_results.columns else 0
        n_output_tokens = int(df_with_results[output_col_name].sum()) if output_col_name in df_with_results.columns else 0
        
        try:
            cost = CostEstimator.estimate_cost(model_name, n_input_tokens, n_output_tokens)
            logger.info(f"Estimated LLM cost: ${cost:.4f} (input: {n_input_tokens} tokens, output: {n_output_tokens} tokens, model: {model_name})")
        except Exception as ce:
            logger.error(f"Could not estimate cost: {ce}")
            
        try:
            log_token_usage_event(
                api_key=api_key,
                model=model_name,
                input_tokens=n_input_tokens,
                output_tokens=n_output_tokens,
                timestamp=None,
                request_id=None
            )
            logger.info("Token usage event logged to output/token_usage_events.jsonl.")
        except Exception as log_exc:
            logger.error(f"[Token Logging Error] Failed to log token usage event: {log_exc}")
    except Exception as cost_exception:
        logger.error(f"[Cost Estimation Error] {cost_exception}")


def process_file_common(
    filepath: str, 
    output_dir: str, 
    config: Dict[str, Any], 
    system_prompt_content: str, 
    response_field: str, 
    encoder: Any, 
    token_limit: int
) -> Tuple[bool, Optional[pd.DataFrame]]:
    """
    Common processing logic for both sequential and concurrent file processing.
    
    Args:
        filepath: Path to the input file
        output_dir: Directory for output files
        config: Configuration dictionary
        system_prompt_content: System prompt content
        response_field: Response field name
        encoder: Tokenizer encoder
        token_limit: Maximum allowed tokens
        
    Returns:
        Tuple containing:
            - Boolean indicating if processing was successful
            - DataFrame with results (if successful) or None (if failed)
    """
    logger.info(f"Processing file: {filepath}")
    
    try:
        df = load_data(filepath)
        logger.info(f"Loaded {len(df)} rows from {filepath}")
        
        if df.empty:
            logger.info(f"No data loaded from {filepath}. Skipping.")
            return False, None
            
        # Check if token limits are respected
        is_under_limit, token_stats = check_token_limits(
            df, system_prompt_content, response_field, encoder, token_limit
        )
        
        if not is_under_limit:
            try:
                llm_client = LLMClient()
                update_token_log(llm_client.api_key, 0)
            except Exception as e:
                logger.error(f"[WARN] Could not log token usage: {e}")
            return False, None
            
        # Log token usage
        try:
            llm_client = LLMClient()
            update_token_log(llm_client.api_key, int(token_stats['total']))
        except Exception as e:
            logger.error(f"[WARN] Could not log token usage: {e}")
            
        # Check batch size limit
        if len(df) > MAX_BATCH_SIZE:
            logger.warning(f"[WARN] Input file contains {len(df)} rows. Only the first {MAX_BATCH_SIZE} will be sent to the API (limit is {MAX_BATCH_SIZE:,} per batch).")
            df = df.iloc[:MAX_BATCH_SIZE].copy()
            
        # Determine whether to use concurrent processing
        force_chunk_count = config.get('force_chunk_count', 0)
        use_concurrent = (force_chunk_count > 1 or 
                         config.get('split_token_limit', 500_000) < token_stats['total'])
        
        # Process the file
        df_with_results = None
        if use_concurrent:
            from .batch_runner import process_file_concurrently
            df_with_results = process_file_concurrently(
                filepath, config, system_prompt_content, response_field, 
                config.get('openai_model_name', DEFAULT_MODEL), 
                llm_client.api_key, encoder
            )
        else:
            llm_client = LLMClient()
            try:
                df_with_results = llm_client.run_batch_job(
                    df, system_prompt_content, response_field_name=response_field, 
                    base_filename_for_tagging=os.path.basename(filepath)
                )
            except Exception as batch_exc:
                logger.error(f"[ERROR] Batch job failed for {filepath}: {batch_exc}")
                return False, None
                
        # Handle errors in results
        if df_with_results is not None:
            # Calculate and log token usage
            model_name = config.get('openai_model_name', DEFAULT_MODEL)
            calculate_and_log_token_usage(
                df_with_results, system_prompt_content, response_field, 
                encoder, model_name, llm_client.api_key
            )
            
            error_rows = df_with_results['llm_score'].str.contains('Error', case=False)
            if error_rows.any():
                logger.info(f"Total rows with errors: {error_rows.sum()}")
                if error_rows.sum() == len(df_with_results):
                    logger.error(f"[BATCH FAILURE] All rows failed for {filepath}. Halting further processing.")
                    return False, df_with_results
                    
            return True, df_with_results
        else:
            logger.error(f"No results obtained for {filepath}")
            return False, None
            
    except Exception as e:
        logger.error(f"An error occurred while processing {filepath}: {e}", exc_info=True)
        log_basename = os.path.splitext(os.path.basename(filepath))[0] + ".log"
        log_base_path = os.path.join(output_dir, log_basename)
        
        if os.path.exists(log_base_path):
            file_root, file_ext = os.path.splitext(log_basename)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            new_log_filename = f"{file_root}_{timestamp}{file_ext}"
            log_path = os.path.join(output_dir, new_log_filename)
            logger.info(f"Log file already exists. Using new filename: {new_log_filename}")
        else:
            log_path = log_base_path
            
        try:
            with open(log_path, 'w') as f_log:
                f_log.write(f"Failed to process {filepath}.\nError: {e}\n")
                f_log.write(traceback.format_exc())
            logger.info(f"Logged error to {log_path}")
        except Exception as save_err:
            logger.error(f"Double fail: exception in processing and then in saving error log for {filepath}: {save_err}")
            
        return False, None


def process_file_wrapper(filepath: str, output_dir: str, config: Dict[str, Any], 
                        system_prompt_content: str, response_field: str, 
                        encoder: Any, token_limit: int) -> bool:
    """
    Process a single file using the common file processing logic and save results.
    
    Args:
        filepath: Path to the input file
        output_dir: Directory for output files
        config: Configuration dictionary
        system_prompt_content: System prompt content
        response_field: Response field name
        encoder: Tokenizer encoder
        token_limit: Maximum allowed tokens
        
    Returns:
        Boolean indicating if processing was successful
    """
    output_path = prepare_output_path(filepath, output_dir, config)
    
    success, df_with_results = process_file_common(
        filepath, output_dir, config, system_prompt_content, 
        response_field, encoder, token_limit
    )
    
    if success and df_with_results is not None:
        save_data(df_with_results.drop(columns=['custom_id'], errors='ignore'), output_path)
        logger.success(f"Processed {filepath}. Results saved to {output_path}")
        logger.success(f"Total rows successfully processed: {len(df_with_results)}")
        return True
    
    return False 