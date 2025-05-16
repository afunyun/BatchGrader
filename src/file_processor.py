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

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import datetime
import traceback
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, TypeVar, Tuple

import pandas as pd
import logging

from batch_job import BatchJob
from constants import (ARCHIVE_DIR, DEFAULT_MODEL, DEFAULT_RESPONSE_FIELD,
                       DEFAULT_SPLIT_TOKEN_LIMIT, LOG_DIR, MAX_BATCH_SIZE,
                       DEFAULT_EVENT_LOG_PATH, DEFAULT_PRICING_CSV_PATH,
                       DEFAULT_TOKEN_USAGE_LOG_PATH)
from cost_estimator import CostEstimator
from data_loader import load_data, save_data
from file_utils import prune_chunked_dir
from input_splitter import split_file_by_token_limit
from llm_client import LLMClient
from rich_display import RichJobTable
from rich.live import Live
from token_tracker import log_token_usage_event, update_token_log
from token_utils import count_completion_tokens, count_input_tokens, create_token_counter
from exceptions import (
    APIError,
    ChunkingError,
    ConfigurationError,
    DataValidationError,
    FileFormatError,
    FileNotFoundError,
    FilePermissionError,
    FileProcessingError,
    OutputDirectoryError,
    TokenLimitError,
)

logger = logging.getLogger(__name__)

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
        raise_on_error: bool = False) -> Tuple[bool, Dict[str, float]]:
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
        logger.error(error_msg)
        if raise_on_error:
            raise ValueError(error_msg)
        return False, {}

    if not isinstance(system_prompt_content,
                      str) or not system_prompt_content.strip():
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
        token_counter = create_token_counter(system_prompt_content,
                                             response_field, encoder)
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
                "Please reduce your batch size or check your usage.")
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


def prepare_output_path(filepath: str, output_dir_str: str,
                        config: Dict[str, Any]) -> str:
    """
    Prepare the output path for saving results.
    
    Args:
        filepath: Original input file path (string)
        output_dir_str: Directory for output files (string)
        config: Configuration dictionary
        
    Returns:
        Output file path (string) for saving results
        
    Raises:
        OutputDirectoryError: If output directory cannot be created
        FilePermissionError: If there are permission issues
    """
    try:
        p_filepath = Path(filepath)
        p_output_dir = Path(output_dir_str)

        filename = p_filepath.name
        file_root = p_filepath.stem
        file_ext = p_filepath.suffix

        # Special handling for test outputs to go into tests/output
        # This logic might need review if project structure changes significantly
        if 'tests' in p_filepath.resolve().as_posix().lower():
            # Assuming __file__ is .../src/file_processor.py
            project_root = Path(__file__).resolve().parent.parent
            p_output_dir = project_root / 'tests' / 'output'

        p_output_dir.mkdir(parents=True, exist_ok=True)

        config_force_chunk = config.get('force_chunk_count', 0)
        if 'legacy' in file_root.lower():
            out_suffix = '_results'
        elif config_force_chunk and config_force_chunk > 1:
            out_suffix = '_forced_results'
        else:
            out_suffix = '_results'

        output_filename = f"{file_root}{out_suffix}{file_ext}"
        p_output_path = p_output_dir / output_filename

        if p_output_path.exists():
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{file_root}{out_suffix}_{timestamp}{file_ext}"
            p_output_path = p_output_dir / output_filename
            logger.info(
                f"Output file already exists. Using new filename: {output_filename}"
            )

        return str(p_output_path
                   )  # Return as string to maintain original type signature
    except PermissionError as e:
        raise FilePermissionError(
            f"Permission denied when creating output path: {e}")
    except OSError as e:
        raise OutputDirectoryError(f"Failed to create output directory: {e}")


def calculate_and_log_token_usage(df_with_results: pd.DataFrame,
                                  system_prompt_content: str,
                                  response_field: str, encoder: Any,
                                  model_name: str, api_key: str) -> None:
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

        if encoder is not None and (
                input_col_name not in df_with_results.columns
                or output_col_name not in df_with_results.columns):
            df_with_results[input_col_name] = df_with_results.apply(
                lambda row: count_input_tokens(row, system_prompt_content,
                                               response_field, encoder),
                axis=1)
            df_with_results[output_col_name] = df_with_results.apply(
                lambda row: count_completion_tokens(row, encoder), axis=1)

        n_input_tokens = int(df_with_results[input_col_name].sum(
        )) if input_col_name in df_with_results.columns else 0
        n_output_tokens = int(df_with_results[output_col_name].sum(
        )) if output_col_name in df_with_results.columns else 0

        try:
            cost = CostEstimator.estimate_cost(model_name, n_input_tokens,
                                               n_output_tokens)
            logger.info(
                f"Estimated LLM cost: ${cost:.4f} (input: {n_input_tokens} tokens, output: {n_output_tokens} tokens, model: {model_name})"
            )
        except Exception as ce:
            logger.error(f"Could not estimate cost: {ce}")

        try:
            log_token_usage_event(api_key=api_key,
                                  model=model_name,
                                  input_tokens=n_input_tokens,
                                  output_tokens=n_output_tokens,
                                  event_log_path=DEFAULT_EVENT_LOG_PATH,
                                  pricing_csv_path=DEFAULT_PRICING_CSV_PATH,
                                  timestamp=None,
                                  request_id=None)
            logger.info(
                "Token usage event logged to output/token_usage_events.jsonl.")
        except Exception as log_exc:
            logger.error(
                f"[Token Logging Error] Failed to log token usage event: {log_exc}"
            )
    except Exception as cost_exception:
        logger.error(f"[Cost Estimation Error] {cost_exception}")


def process_file_common(
        filepath: str, output_dir: str, config: Dict[str, Any],
        system_prompt_content: str, response_field: str, encoder: Any,
        token_limit: int) -> Tuple[bool, Optional[pd.DataFrame]]:
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
            
    Raises:
        FileNotFoundError: If input file is not found
        FileFormatError: If file format is invalid
        DataValidationError: If data validation fails
        TokenLimitError: If token limits are exceeded
        APIError: If API calls fail
    """
    logger.info(f"Processing file: {filepath}")

    try:
        # Convert filepath to Path for internal use, but keep original string for messages if needed
        p_filepath = Path(filepath)
        if not p_filepath.exists():
            raise FileNotFoundError(f"Input file not found: {filepath}")

        df = load_data(filepath)
        if df is None:
            raise FileFormatError(f"Failed to load data from {filepath}")

        logger.info(f"Loaded {len(df)} rows from {filepath}")

        if df.empty:
            logger.info(f"No data loaded from {filepath}. Skipping.")
            return False, None

        # Check if token limits are respected
        is_under_limit, token_stats = check_token_limits(
            df, system_prompt_content, response_field, encoder, token_limit)

        if not is_under_limit:
            try:
                llm_client = LLMClient()
                update_token_log(llm_client.api_key,
                                 0,
                                 log_path=DEFAULT_TOKEN_USAGE_LOG_PATH)
            except Exception as e:
                logger.error(f"[WARN] Could not log token usage: {e}")
            raise TokenLimitError(f"Token limit exceeded for {filepath}")

        # Log token usage
        try:
            llm_client = LLMClient()
            update_token_log(llm_client.api_key,
                             int(token_stats['total']),
                             log_path=DEFAULT_TOKEN_USAGE_LOG_PATH)
        except Exception as e:
            logger.error(f"[WARN] Could not log token usage: {e}")

        # Check batch size limit
        if len(df) > MAX_BATCH_SIZE:
            logger.warning(
                f"[WARN] Input file contains {len(df)} rows. Only the first {MAX_BATCH_SIZE} will be sent to the API (limit is {MAX_BATCH_SIZE:,} per batch)."
            )
            df = df.iloc[:MAX_BATCH_SIZE].copy()

        # Determine whether to use concurrent processing
        force_chunk_count = config.get('force_chunk_count', 0)
        use_concurrent = (force_chunk_count > 1 or config.get(
            'split_token_limit', 500_000) < token_stats['total'])

        # Process the file
        df_with_results = None
        if use_concurrent:
            df_with_results = process_file_concurrently(
                filepath, config, system_prompt_content, response_field,
                config.get('openai_model_name', DEFAULT_MODEL),
                llm_client.api_key, encoder)
        else:
            llm_client = LLMClient()
            try:
                df_with_results = llm_client.run_batch_job(
                    df,
                    system_prompt_content,
                    response_field_name=response_field,
                    base_filename_for_tagging=os.path.basename(filepath))
            except Exception as batch_exc:
                raise APIError(f"Batch job failed for {filepath}: {batch_exc}")

        # Handle errors in results
        if df_with_results is not None:
            # Calculate and log token usage
            model_name = config.get('openai_model_name', DEFAULT_MODEL)
            calculate_and_log_token_usage(df_with_results,
                                          system_prompt_content,
                                          response_field, encoder, model_name,
                                          llm_client.api_key)

            error_rows = df_with_results['llm_score'].str.contains('Error',
                                                                   case=False)
            if error_rows.any():
                logger.info(f"Total rows with errors: {error_rows.sum()}")
                if error_rows.sum() == len(df_with_results):
                    raise DataValidationError(
                        f"All rows failed for {filepath}. Halting further processing."
                    )
                    return False, df_with_results

            return True, df_with_results
        else:
            raise DataValidationError(f"No results obtained for {filepath}")

    except (FileNotFoundError, FileFormatError, DataValidationError,
            TokenLimitError, APIError) as e:
        # Re-raise known exceptions
        raise
    except Exception as e:
        # Convert unknown exceptions to FileProcessingError
        raise FileProcessingError(
            f"An unexpected error occurred while processing {filepath}: {e}"
        ) from e


def process_file_wrapper(filepath: str, output_dir: str, config: Dict[str,
                                                                      Any],
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
    try:
        output_path = prepare_output_path(filepath, output_dir, config)

        success, df_with_results = process_file_common(filepath, output_dir,
                                                       config,
                                                       system_prompt_content,
                                                       response_field, encoder,
                                                       token_limit)

        if success and df_with_results is not None:
            # Only drop custom_id if it was generated by us (not from original data)
            original_df = load_data(filepath)
            if 'custom_id' not in original_df.columns:
                df_with_results = df_with_results.drop(columns=['custom_id'],
                                                       errors='ignore')
                logger.debug(
                    f"Dropped generated 'custom_id' column from results for {filepath}"
                )

            save_data(df_with_results, output_path)
            logger.success(
                f"Processed {filepath}. Results saved to {output_path}")
            logger.success(
                f"Total rows successfully processed: {len(df_with_results)}")
            return True

        return False

    except (FileNotFoundError, FileFormatError, DataValidationError,
            TokenLimitError, APIError, OutputDirectoryError,
            FilePermissionError) as e:
        logger.error(f"[ERROR] {str(e)}")
        return False
    except Exception as e:
        logger.error(
            f"[CRITICAL ERROR] Unexpected error in process_file_wrapper: {e}",
            exc_info=True)
        return False


def _generate_chunk_job_objects(
    original_filepath: str,
    system_prompt_content: str,
    config: dict,
    tiktoken_encoding_func: Any,  # tiktoken.Encoding or similar
    response_field: str,
    llm_model_name: Optional[str],
    api_key_prefix: Optional[str]  # Typically from llm_client.api_key_prefix
) -> list[BatchJob]:
    """Splits the input file using input_splitter.split_file and creates BatchJob objects for each chunk."""

    splitter_config = config.get('input_splitter_options', {})
    default_split_limit = config.get('split_token_limit',
                                     DEFAULT_SPLIT_TOKEN_LIMIT)
    max_tokens_per_chunk = splitter_config.get('max_tokens_per_chunk',
                                               default_split_limit)
    max_rows_per_chunk = splitter_config.get(
        'max_rows_per_chunk', config.get('split_row_limit', None))
    force_chunk_count_val = splitter_config.get('force_chunk_count', None)

    logger.debug(f"Generating chunk job objects for: {original_filepath}")
    logger.debug(
        f"Using splitter config: max_tokens={max_tokens_per_chunk}, max_rows={max_rows_per_chunk}, force_chunks={force_chunk_count_val}"
    )

    if not hasattr(tiktoken_encoding_func, 'encode'):
        logger.error(
            f"Invalid tiktoken_encoding_func passed. Type: {type(tiktoken_encoding_func)}. Attempting to load default."
        )
        try:
            import tiktoken  # Local import for safety
            tiktoken_encoding_func = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.critical(
                f"Failed to load default tiktoken encoder: {e}. Chunking will likely fail.",
                exc_info=True)
            return []

    def _count_row_tokens(row: pd.Series) -> int:  # Type hint for row
        row_content = " ".join(
            str(value) for value in row.values if pd.notna(value))
        full_content_for_tokenization = system_prompt_content + "\n" + row_content
        return len(
            tiktoken_encoding_func.encode(full_content_for_tokenization))

    try:
        chunk_file_paths, _ = split_file_by_token_limit(
            input_path=original_filepath,
            token_limit=max_tokens_per_chunk,
            count_tokens_fn=_count_row_tokens,
            response_field=response_field,
            row_limit=max_rows_per_chunk,
            force_chunk_count=force_chunk_count_val,
            logger_override=logger)
    except Exception as e:
        logger.error(
            f"Error during file splitting for {original_filepath}: {e}",
            exc_info=True)
        return []

    if not chunk_file_paths:
        logger.warning(
            f"No chunk files generated by input_splitter for {original_filepath}. The file might be empty or unreadable."
        )
        return []

    jobs = []
    for i, chunk_path_str in enumerate(chunk_file_paths):
        chunk_path = Path(chunk_path_str)
        try:
            logger.debug(f"Loading chunk file: {chunk_path}")
            chunk_df = load_data(str(chunk_path))
            if chunk_df is None or chunk_df.empty:
                logger.warning(
                    f"Skipping empty or unreadable chunk file: {chunk_path}")
                continue

            # Handle ID column logic
            if 'custom_id' in chunk_df.columns and 'id' in chunk_df.columns:
                logger.info(
                    f"Both 'id' and 'custom_id' columns found in chunk: {chunk_path.name}. Using existing 'custom_id' column."
                )
            elif 'id' in chunk_df.columns:
                chunk_df = chunk_df.rename(columns={'id': 'custom_id'})
                logger.info(
                    f"Renamed 'id' to 'custom_id' for chunk: {chunk_path.name}"
                )

            # Ensure custom_id is string type
            if 'custom_id' in chunk_df.columns:
                if not pd.api.types.is_string_dtype(chunk_df['custom_id']):
                    logger.debug(
                        f"Casting 'custom_id' to string for chunk: {chunk_path.name} (current type: {chunk_df['custom_id'].dtype})"
                    )
                    chunk_df['custom_id'] = chunk_df['custom_id'].astype(str)
            else:
                logger.warning(
                    f"Neither 'custom_id' nor 'id' column found in chunk: {chunk_path.name}. This might cause issues if LLMClient requires an ID."
                )

            job = BatchJob(chunk_id_str=chunk_path.stem,
                           chunk_df=chunk_df,
                           system_prompt=system_prompt_content,
                           response_field=response_field,
                           original_filepath=original_filepath,
                           chunk_file_path=str(chunk_path),
                           llm_model=llm_model_name,
                           api_key_prefix=api_key_prefix,
                           status="pending")
            jobs.append(job)
        except Exception as e:
            logger.error(
                f"Error processing chunk file {chunk_path} into BatchJob: {e}",
                exc_info=True)
            failed_job = BatchJob(
                chunk_id_str=chunk_path.stem
                if chunk_path else f"failed_chunk_preparation_{i}",
                chunk_df=None,
                system_prompt=system_prompt_content,
                response_field=response_field,
                original_filepath=original_filepath,
                chunk_file_path=str(chunk_path) if chunk_path else None,
                llm_model=llm_model_name,
                api_key_prefix=api_key_prefix,
                status="error",
                error_message=
                f"Failed to load/prepare BatchJob from chunk: {e}",
                error_details=traceback.format_exc())
            jobs.append(failed_job)

    logger.info(
        f"Generated {len(jobs)} BatchJob objects from {original_filepath}.")
    return jobs


def _execute_single_batch_job_task(
        batch_job: BatchJob,
        llm_client: Optional[LLMClient] = None,
        response_field_name: str = "response",
        config: Optional[Dict[str, Any]] = None) -> BatchJob:
    """
    Worker function to process a single BatchJob chunk.
    Updates batch_job status and results in place. Returns the updated batch_job object.
    
    Args:
        batch_job: The BatchJob to process
        llm_client: Optional LLMClient instance. If None, creates a new one for thread safety
        response_field_name: Name of response field to process
        config: Optional configuration dictionary to pass to LLMClient if created
        
    Returns:
        The updated BatchJob object
    """
    if batch_job.chunk_df is None or batch_job.chunk_df.empty:
        if batch_job.status != "error":
            batch_job.status = "error"
            batch_job.error_message = "Chunk DataFrame is None or empty at task execution."
            batch_job.error_details = "Chunk DataFrame was not loaded or was empty when task started."
            logger.error(
                f"[{batch_job.chunk_id_str}] Skipping task execution: {batch_job.error_message}"
            )
        return batch_job

    # Create a new LLMClient if one wasn't provided for thread safety
    if llm_client is None:
        try:
            llm_client = LLMClient(config=config)
            logger.debug(
                f"[{batch_job.chunk_id_str}] Created new LLMClient instance for thread safety"
            )
        except Exception as e:
            batch_job.status = "error"
            batch_job.error_message = f"Failed to create LLMClient: {str(e)}"
            batch_job.error_details = traceback.format_exc()
            logger.error(
                f"[{batch_job.chunk_id_str}] {batch_job.error_message}")
            return batch_job

    try:
        batch_job.status = "running"
        logger.info(f"[{batch_job.chunk_id_str}] Task starting execution.")

        if 'custom_id' not in batch_job.chunk_df.columns:
            logger.warning(
                f"[{batch_job.chunk_id_str}] 'custom_id' column is missing from chunk_df at the start of _execute_single_batch_job_task."
            )

        api_result = llm_client.run_batch_job(
            batch_job.chunk_df,
            batch_job.system_prompt,
            response_field_name=response_field_name,
            base_filename_for_tagging=batch_job.chunk_id_str)

        if isinstance(api_result, pd.DataFrame):
            batch_job.result_data = api_result
            batch_job.status = "completed"
            logger.debug(
                f"[{batch_job.chunk_id_str}] Task completed, result is DataFrame with {len(api_result)} rows."
            )
        elif isinstance(api_result,
                        dict) and ('error' in api_result or
                                   'custom_id_of_failed_item' in api_result):
            batch_job.status = "failed"
            batch_job.error_message = api_result.get(
                'error_message', api_result.get('error', str(api_result)))
            batch_job.error_details = api_result
            batch_job.result_data = None
            logger.warning(
                f"[{batch_job.chunk_id_str}] Task resulted in API failure: {batch_job.error_message}"
            )
        else:
            batch_job.status = "error"
            batch_job.error_message = f"Unexpected API result type: {type(api_result)}"
            batch_job.error_details = str(api_result)
            batch_job.result_data = None
            logger.error(
                f"[{batch_job.chunk_id_str}] Task failed with unexpected API result: {batch_job.error_message}"
            )

    except Exception as exc:
        logger.error(
            f"[{batch_job.chunk_id_str}] Exception in _execute_single_batch_job_task: {exc}",
            exc_info=True)
        batch_job.status = "failed"
        batch_job.error_message = str(exc)
        batch_job.error_details = traceback.format_exc()
        batch_job.result_data = None

    logger.info(
        f"[{batch_job.chunk_id_str}] Task finished with status: {batch_job.status}"
    )
    return batch_job


def _pfc_submit_jobs(
        jobs_to_submit: List[BatchJob],
        response_field_name: str,
        max_workers_config: int,
        config: Optional[Dict[str, Any]] = None) -> Dict[Any, BatchJob]:
    """Submits BatchJob objects to a ThreadPoolExecutor for concurrent processing.

    Each job is processed by `_execute_single_batch_job_task`. A new LLMClient instance
    is intended to be created within each task execution for thread safety, using the provided `config`.

    Args:
        jobs_to_submit: A list of `BatchJob` objects to be processed.
        response_field_name: The name of the field in the DataFrame where the LLM response should be stored/read from.
        max_workers_config: The maximum number of worker threads for the ThreadPoolExecutor.
        config: Optional configuration dictionary passed to `_execute_single_batch_job_task` for LLMClient instantiation.

    Returns:
        A dictionary mapping Future objects to their corresponding BatchJob objects.
    """
    with ThreadPoolExecutor(max_workers=max_workers_config) as executor:
        # Submit each job with its own LLMClient instance
        # Don't create LLMClient outside of the executor to avoid sharing between threads
        future_to_job_map = {}
        for job in jobs_to_submit:
            # Each submit call will create a separate LLMClient inside the _execute_single_batch_job_task
            # Pass config to ensure consistent configuration
            future = executor.submit(_execute_single_batch_job_task, job, None,
                                     response_field_name, config)
            future_to_job_map[future] = job

    logger.info(f"Submitted {len(future_to_job_map)} chunk jobs to executor.")
    return future_to_job_map


def _pfc_process_completed_future(future: Any,
                                  future_to_job_map: Dict[Any, BatchJob],
                                  completed_jobs_list: List[BatchJob],
                                  live_display: Optional[Live],
                                  rich_table: RichJobTable,
                                  all_jobs_list: List[BatchJob],
                                  halt_on_failure_flag: bool,
                                  original_filepath: str,
                                  llm_output_column_name: str) -> bool:
    """Processes a single completed future from the concurrent execution pool."""
    job_from_future = future_to_job_map[future]
    logger.info(
        f"Future completed for chunk {job_from_future.chunk_id_str}. Processing result..."
    )

    try:
        # Retrieve the result from the future (the BatchJob object updated by the task)
        processed_job_in_task = future.result()

        # Update the original job object in all_jobs_list with the results from the task
        job_from_future.status = processed_job_in_task.status
        job_from_future.error_message = processed_job_in_task.error_message
        job_from_future.error_details = processed_job_in_task.error_details
        job_from_future.result_data = processed_job_in_task.result_data

        logger.info(
            f"[{job_from_future.chunk_id_str}] Processed job status from task: {job_from_future.status}"
        )
        if job_from_future.result_data is not None:
            logger.debug(
                f"[{job_from_future.chunk_id_str}] Result data from task (type {type(job_from_future.result_data)}): Preview: {str(job_from_future.result_data.head(1))[:200] if isinstance(job_from_future.result_data, pd.DataFrame) else str(job_from_future.result_data)[:200]}"
            )
        else:
            logger.debug(
                f"[{job_from_future.chunk_id_str}] Result data from task is None."
            )

        if job_from_future.status == "completed":
            if isinstance(job_from_future.result_data, pd.DataFrame):
                completed_jobs_list.append(job_from_future)
                logger.success(
                    f"Chunk {job_from_future.chunk_id_str} completed successfully. DataFrame stored."
                )
            else:
                # Handle case where status is 'completed' but result_data is not a DataFrame
                job_from_future.status = "error"
                job_from_future.error_message = f"Completed status but result_data is not DataFrame (type: {type(job_from_future.result_data)})"
                logger.error(
                    f"Chunk {job_from_future.chunk_id_str}: {job_from_future.error_message}"
                )
                # Create a placeholder error DataFrame for this job
                error_df_data = {
                    'custom_id': job_from_future.chunk_id_str,
                    llm_output_column_name:
                    f"ERROR: {job_from_future.error_message}",
                    'error_type': 'DataFrameTypeError',
                    'original_file': original_filepath,
                    'chunk_id': job_from_future.chunk_id_str
                }
                job_from_future.result_data = pd.DataFrame([error_df_data])
                completed_jobs_list.append(job_from_future)

        # Handle failed or error status cases by creating a placeholder error DataFrame
        if job_from_future.status == "failed" or job_from_future.status == "error":
            logger.error(
                f"Chunk {job_from_future.chunk_id_str} reported as {job_from_future.status}. Error: {job_from_future.error_message}"
            )

            # Ensure an error DataFrame is created and added to results if not already processed
            if job_from_future not in completed_jobs_list:
                error_custom_id = None
                # Attempt to get a custom_id from error details if available
                if isinstance(job_from_future.error_details, dict):
                    error_custom_id = job_from_future.error_details.get(
                        'custom_id',
                        job_from_future.error_details.get(
                            'custom_id_of_failed_item'))

                # Fallback to chunk_id_str if no specific custom_id found in error details
                if error_custom_id is None and job_from_future.chunk_df is not None and 'custom_id' in job_from_future.chunk_df.columns:
                    # This case might be less common if error occurred before df processing
                    error_custom_id = job_from_future.chunk_id_str
                elif error_custom_id is None:
                    error_custom_id = job_from_future.chunk_id_str  # Default to job's own chunk_id

                error_df_data = {
                    'custom_id': error_custom_id,
                    llm_output_column_name:
                    f"ERROR: {job_from_future.error_message}",
                    'error_type':
                    job_from_future.status.capitalize() + 'Error',
                    'original_file': original_filepath,
                    'chunk_id': job_from_future.chunk_id_str
                }
                if job_from_future.error_details:
                    error_df_data['error_details'] = str(
                        job_from_future.error_details)

                job_from_future.result_data = pd.DataFrame([error_df_data])
                completed_jobs_list.append(job_from_future)
                logger.warning(
                    f"Chunk {job_from_future.chunk_id_str} {job_from_future.status}, but continuing. Error info DataFrame created and added to results."
                )

            # If halt_on_failure is True, signal to stop processing further jobs for this file
            if halt_on_failure_flag:
                logger.error(
                    f"[HALT] {job_from_future.status.capitalize()} detected in chunk {job_from_future.chunk_id_str}. Halting processing for {Path(original_filepath).name}."
                )
                if live_display:
                    live_display.update(rich_table.build_table(all_jobs_list))
                return True  # Signal to halt

    except Exception as e:
        # Catch any unexpected exception during future processing
        job_from_future.status = "error"
        job_from_future.error_message = f"Exception processing future for {job_from_future.chunk_id_str}: {e}"
        job_from_future.error_details = traceback.format_exc()
        logger.error(
            f"[{job_from_future.chunk_id_str}] Exception in _pfc_process_completed_future: {e}",
            exc_info=True)

        if halt_on_failure_flag:
            logger.error(
                f"[HALT] Future processing error for {job_from_future.chunk_id_str}. Halting processing for {Path(original_filepath).name}."
            )
            if live_display:
                live_display.update(rich_table.build_table(all_jobs_list))
            return True  # Signal to halt
        else:
            # If not halting, create an error DataFrame for this job
            error_df = pd.DataFrame([{
                'custom_id': job_from_future.chunk_id_str,
                llm_output_column_name:
                f"ERROR: {job_from_future.error_message}",
                'error_type': 'FutureProcessingError',
                'original_file': original_filepath,
                'chunk_id': job_from_future.chunk_id_str,
                'error_details': traceback.format_exc()
            }])
            job_from_future.result_data = error_df
            completed_jobs_list.append(job_from_future)
            logger.warning(
                f"Chunk {job_from_future.chunk_id_str} had future processing error, but continuing. Error info DataFrame created."
            )

    # Update the live display with the latest job statuses
    if live_display: live_display.update(rich_table.build_table(all_jobs_list))

    # Check again for halt condition after processing, in case status was set to failed/error inside try block
    if (job_from_future.status == "failed"
            or job_from_future.status == "error") and halt_on_failure_flag:
        logger.debug(
            f"Confirming halt for {Path(original_filepath).name} due to chunk {job_from_future.chunk_id_str} status {job_from_future.status}."
        )
        return True  # Signal to halt

    return False  # No halt signal


def _pfc_aggregate_and_cleanup(
        completed_jobs_list: List[BatchJob], original_filepath: str,
        response_field_name: str) -> Optional[pd.DataFrame]:
    """Aggregates results from completed BatchJob objects and cleans up chunked files."""
    all_results_dfs = []
    total_processed_rows_from_completed = 0
    original_file_path = Path(original_filepath)

    for job in completed_jobs_list:
        if job.result_data is not None and isinstance(
                job.result_data, pd.DataFrame) and not job.result_data.empty:
            all_results_dfs.append(job.result_data)
            if job.status == "completed":
                total_processed_rows_from_completed += len(job.result_data)
        elif job.status == "completed" and (job.result_data is None
                                            or job.result_data.empty):
            logger.warning(
                f"Job {job.chunk_id_str} status is 'completed' but has no result_data. Skipping aggregation for this job."
            )

    if not all_results_dfs:
        logger.error(
            f"[FAILURE] No valid results (DataFrames) to aggregate for {original_file_path.name}. No combined output file will be produced."
        )
        chunked_dir_path = original_file_path.parent / '_chunked'
        if chunked_dir_path.exists():
            prune_chunked_dir(str(chunked_dir_path))
        return None

    try:
        combined_df = pd.concat(all_results_dfs, ignore_index=True)
        logger.success(
            f"Results aggregated for {original_file_path.name}. Total rows in combined output: {len(combined_df)} ({total_processed_rows_from_completed} rows from successfully completed chunks)."
        )

        # Handle ID column mapping
        if 'custom_id' in combined_df.columns:
            if 'id' not in combined_df.columns:
                combined_df['id'] = combined_df['custom_id']
                logger.debug(
                    f"Added 'id' column to aggregated results for {original_file_path.name}, copied from 'custom_id'."
                )
            else:
                logger.debug(
                    f"Both 'id' and 'custom_id' columns exist in aggregated results for {original_file_path.name}. Keeping both."
                )

    except Exception as e:
        logger.error(
            f"Failed to concatenate results for {original_file_path.name}: {e}",
            exc_info=True)
        chunked_dir_path = original_file_path.parent / '_chunked'
        if chunked_dir_path.exists(): prune_chunked_dir(str(chunked_dir_path))
        return None

    chunked_dir_path = original_file_path.parent / '_chunked'
    if chunked_dir_path.exists():
        prune_chunked_dir(str(chunked_dir_path))
        logger.info(f"Cleaned up chunk directory: {chunked_dir_path}")

    return combined_df


def process_file_concurrently(
        filepath: str, config: Dict[str, Any], system_prompt_content: str,
        response_field: str, llm_model_name: str, api_key_prefix: str,
        tiktoken_encoding_func: Any) -> Optional[pd.DataFrame]:
    """
    Process a file concurrently using multiple threads.

    Args:
        filepath: Path to the input file
        config: Configuration dictionary
        system_prompt_content: System prompt content
        response_field: Response field name
        llm_model_name: Name of the LLM model to use
        api_key_prefix: API key prefix for token tracking
        tiktoken_encoding_func: Function to get tiktoken encoding

    Returns:
        DataFrame with results if successful, None if failed
    """
    logger.info(f"Starting concurrent processing for: {filepath}")

    try:
        jobs: List[BatchJob] = _generate_chunk_job_objects(
            original_filepath=filepath,
            config=config,
            system_prompt_content=system_prompt_content,
            response_field=response_field,
            llm_model_name=llm_model_name,
            api_key_prefix=api_key_prefix,
            tiktoken_encoding_func=tiktoken_encoding_func)

        if not jobs:
            logger.warning(
                f"No BatchJob objects generated for {filepath}. Cannot proceed "
                "with concurrent processing.")
            return None

        halt_on_failure = config.get('halt_on_chunk_failure', True)
        completed_jobs: List[BatchJob] = []

        rich_table = RichJobTable()
        failure_detected_and_halted = False

        max_workers = config.get('max_simultaneous_batches',
                                 config.get('max_workers', 2))
        future_to_job_map: Dict[Any, BatchJob] = _pfc_submit_jobs(
            jobs, response_field, max_workers, config)

        try:
            llm_actual_output_column_name = config.get(
                'llm_output_column_name', response_field)

            with Live(rich_table.build_table(jobs),
                      console=rich_table.console,
                      refresh_per_second=4,
                      vertical_overflow="visible") as live:
                for future in as_completed(future_to_job_map.keys()):
                    should_halt = _pfc_process_completed_future(
                        future=future,
                        future_to_job_map=future_to_job_map,
                        completed_jobs_list=completed_jobs,
                        live_display=live,
                        rich_table=rich_table,
                        all_jobs_list=jobs,
                        halt_on_failure_flag=halt_on_failure,
                        original_filepath=filepath,
                        llm_output_column_name=llm_actual_output_column_name)
                    if should_halt:
                        failure_detected_and_halted = True
                        logger.error(
                            f"Halt signal received while processing {filepath}. Cancelling remaining jobs."
                        )
                        break

                if failure_detected_and_halted:
                    cancelled_count = 0
                    for fut_to_cancel, job_to_cancel in future_to_job_map.items(
                    ):
                        if not fut_to_cancel.done():
                            if fut_to_cancel.cancel():
                                cancelled_count += 1
                                job_to_cancel.status = "cancelled"
                                logger.debug(
                                    f"Marked job {job_to_cancel.chunk_id_str} as cancelled."
                                )
                    if cancelled_count > 0:
                        logger.warning(
                            f"Cancelled {cancelled_count} pending chunk jobs for {filepath} due to halt on failure."
                        )
                    live.update(rich_table.build_table(jobs))

            logger.info(
                f"All futures processed for {filepath}. Aggregating results..."
            )
            final_df = _pfc_aggregate_and_cleanup(completed_jobs, filepath,
                                                  response_field)
            if final_df is not None:
                logger.success(
                    f"Concurrent processing of {filepath} completed. Aggregated DataFrame has {len(final_df)} rows."
                )
            else:
                logger.error(
                    f"Concurrent processing of {filepath} failed to produce an aggregated DataFrame."
                )
            return final_df

        except Exception as e:
            logger.error(f"Error during concurrent processing: {str(e)}")
            return None

    except Exception as e:
        logger.error(f"Error during concurrent processing: {str(e)}")
        return None
