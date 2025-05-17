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

import datetime
import logging
import os
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from rich.live import Live

from batchgrader.batch_job import BatchJob
from batchgrader.constants import (
    DEFAULT_EVENT_LOG_PATH,
    DEFAULT_MODEL,
    DEFAULT_PRICING_CSV_PATH,
    DEFAULT_SPLIT_TOKEN_LIMIT,
    DEFAULT_TOKEN_USAGE_LOG_PATH,
    MAX_BATCH_SIZE,
)
from batchgrader.cost_estimator import CostEstimator
from batchgrader.data_loader import load_data, save_data
from batchgrader.exceptions import (
    APIError,
    BatchGraderFileNotFoundError,
    DataValidationError,
    FileFormatError,
    FilePermissionError,
    FileProcessingError,
    OutputDirectoryError,
    TokenLimitError,
)
from batchgrader.file_utils import prune_chunked_dir
from batchgrader.input_splitter import split_file_by_token_limit
from batchgrader.llm_client import LLMClient
from batchgrader.rich_display import RichJobTable
from batchgrader.token_tracker import log_token_usage_event, update_token_log
from batchgrader.token_utils import count_completion_tokens, count_input_tokens
from batchgrader.token_utils import create_token_counter as create_token_counter_util

# Re-expose for test compatibility or if other modules expect it directly here
create_token_counter = create_token_counter_util

logger = logging.getLogger(__name__)


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
            "input_path": self.input_path,
            "output_path": str(self.output_path) if self.output_path else None,
            "rows_processed": self.rows_processed,
            "token_usage": self.token_usage,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.duration,
            "error": str(self.error) if self.error else None,
        }


def _handle_dataframe_error(
    error_msg: str, raise_on_error: bool, logger_instance: logging.Logger
):
    logger_instance.error(error_msg)
    if raise_on_error:
        raise ValueError(error_msg)
    return False, {}


def _handle_validation_error(
    error_msg: str, raise_on_error: bool, logger_instance: logging.Logger
):
    logger_instance.error(error_msg)
    if raise_on_error:
        raise ValueError(error_msg)
    return False, {}


def check_token_limits(
    df: pd.DataFrame,
    system_prompt_content: str,
    response_field: str,
    encoder: Any,
    token_limit: int,
    raise_on_error: bool = False,
    filepath: Optional[str] = None,
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
        filepath: Optional path of the file being checked, for error messages.

    Returns:
        Tuple containing:
            - Boolean indicating if tokens are under limit
            - Dictionary with token statistics (total, average, max)

    Raises:
        ValueError: If input validation fails and raise_on_error is True.
        TypeError: If input types are incorrect and raise_on_error is True.
        TokenLimitError: If token limits are exceeded and raise_on_error is True.
        RuntimeError: For other unexpected errors during token checking if raise_on_error is True.
    """

    def _validate_inputs():
        if not isinstance(df, pd.DataFrame):
            error_msg = f"df must be a pandas DataFrame, got {type(df).__name__}"
            if raise_on_error:
                raise TypeError(error_msg)
            logger.error(error_msg)
            return False, {}

        if df.empty:
            return _handle_dataframe_error(
                "DataFrame cannot be empty", raise_on_error, logger
            )

        if (
            not isinstance(system_prompt_content, str)
            or not system_prompt_content.strip()
        ):
            return _handle_validation_error(
                "system_prompt_content must be a non-empty string",
                raise_on_error,
                logger,
            )

        if not isinstance(response_field, str) or not response_field.strip():
            return _handle_validation_error(
                "response_field must be a non-empty string", raise_on_error, logger
            )

        if response_field not in df.columns:
            return _handle_validation_error(
                f"response_field '{response_field}' not found in DataFrame columns",
                raise_on_error,
                logger,
            )

        if not isinstance(token_limit, int) or token_limit <= 0:
            return _handle_validation_error(
                f"token_limit must be a positive integer, got {token_limit}",
                raise_on_error,
                logger,
            )

        if encoder is None:
            return _handle_validation_error(
                "encoder cannot be None. It must be a tiktoken.Encoding or a callable.",
                raise_on_error,
                logger,
            )
        return None

    def _log_token_usage(tokens_submitted, tokens_used_for_error=False):
        try:
            llm_client = LLMClient()
            update_token_log(
                api_key=llm_client.api_key,
                tokens_submitted=tokens_submitted,
                tokens_used_for_error=tokens_used_for_error,
                log_path=DEFAULT_TOKEN_USAGE_LOG_PATH,
            )
        except Exception as e:
            logger.error(f"[WARN] Could not log token usage detail: {e}")

    validation_result = _validate_inputs()
    if validation_result is not None:
        return validation_result

    try:
        token_counter_func = create_token_counter_util(
            system_prompt_content, response_field, encoder
        )
        token_counts = df.apply(token_counter_func, axis=1)

        if token_counts.empty:
            return True, {"total": 0, "average": 0, "max": 0}

        token_stats = {
            "total": float(token_counts.sum()),
            "average": float(token_counts.mean()),
            "max": float(token_counts.max()),
        }

        is_under_limit = token_stats["total"] <= token_limit

        logger.debug(
            f"[TOKEN COUNT] Total: {int(token_stats['total'])}, "
            f"Avg: {token_stats['average']:.1f}, "
            f"Max: {int(token_stats['max'])}"
        )

        if not is_under_limit:
            _log_token_usage(0, tokens_used_for_error=True)
            error_location = f"for {filepath}" if filepath else "for an input dataframe"
            if not raise_on_error:
                return False, token_stats

            details = (
                f"Details: {token_stats['total']:.0f} tokens "
                f"({token_stats['average']:.1f} avg, {token_stats['max']:.0f} max) "
                f"vs limit {token_limit:.0f}."
            )
            raise TokenLimitError(f"Token limit exceeded {error_location}. {details}")

        _log_token_usage(int(token_stats["total"]))
        return is_under_limit, token_stats

    except TokenLimitError:
        raise
    except Exception as e:
        error_msg = f"Error checking token limits: {str(e)}"
        logger.error(error_msg, exc_info=True)
        if raise_on_error:
            raise RuntimeError(error_msg) from e
        return False, {}


def prepare_output_path(
    filepath: str,
    output_dir_str: str,
    config: Dict[str, Any],
    is_reprocessing_run: bool = False,
) -> str:
    """
    Prepare the output path for saving results.

    Args:
        filepath: Original input file path (string)
        output_dir_str: Directory for output files (string)
        config: Configuration dictionary
        is_reprocessing_run: If True, append "_reprocessed" to the output filename

    Returns:
        Output file path (string) for saving results

    Raises:
        OutputDirectoryError: If output directory cannot be created
        FilePermissionError: If there are permission issues
    """
    try:
        p_filepath = Path(filepath)
        p_output_dir = Path(output_dir_str)

        file_root = p_filepath.stem
        file_ext = p_filepath.suffix

        if "tests" in p_filepath.resolve().as_posix().lower():
            project_root = Path(__file__).resolve().parent.parent
            p_output_dir = project_root / "tests" / "output"

        p_output_dir.mkdir(parents=True, exist_ok=True)

        config_force_chunk = config.get("force_chunk_count", 0)
        if "legacy" in file_root.lower():
            out_suffix = "_results"
        elif config_force_chunk and config_force_chunk > 1:
            out_suffix = "_forced_results"
        else:
            out_suffix = "_results"

        base_name_parts = [file_root]
        if is_reprocessing_run:
            base_name_parts.append("_reprocessed")
        base_name_parts.append(out_suffix)
        final_base_name = "".join(base_name_parts)

        output_filename = f"{final_base_name}{file_ext}"
        p_output_path = p_output_dir / output_filename

        if p_output_path.exists():
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            # Changed from file_root to final_base_name
            output_filename = f"{final_base_name}_{timestamp}{file_ext}"
            p_output_path = p_output_dir / output_filename
            logger.info(
                f"Output file already exists. Using new filename: {output_filename}"
            )

        return str(p_output_path)
    except PermissionError as e:
        raise FilePermissionError(
            f"Permission denied when creating output path: {e}"
        ) from e
    except OSError as e:
        raise OutputDirectoryError(f"Failed to create output directory: {e}") from e


def calculate_and_log_token_usage(
    df_with_results: pd.DataFrame,
    system_prompt_content: str,
    response_field: str,
    encoder: Any,
    model_name: str,
    api_key: str,
) -> None:
    """
    Calculate token usage stats and log to the token tracking system.
    """
    try:
        input_col_name = "input_tokens"
        output_col_name = "output_tokens"

        if encoder is not None and (
            input_col_name not in df_with_results.columns
            or output_col_name not in df_with_results.columns
        ):
            df_with_results[input_col_name] = df_with_results.apply(
                lambda row: count_input_tokens(
                    row, system_prompt_content, response_field, encoder
                ),
                axis=1,
            )
            df_with_results[output_col_name] = df_with_results.apply(
                lambda row: count_completion_tokens(row, encoder), axis=1
            )

        n_input_tokens = (
            int(df_with_results[input_col_name].sum())
            if input_col_name in df_with_results.columns
            else 0
        )
        n_output_tokens = (
            int(df_with_results[output_col_name].sum())
            if output_col_name in df_with_results.columns
            else 0
        )

        try:
            cost = CostEstimator.estimate_cost(
                model_name, n_input_tokens, n_output_tokens
            )
            logger.info(
                f"Estimated LLM cost: ${cost:.4f} (input: {n_input_tokens} tokens, output: {n_output_tokens} tokens, model: {model_name})"
            )
        except Exception as ce:
            logger.error(f"Could not estimate cost: {ce}")

        try:
            log_token_usage_event(
                api_key=api_key,
                model=model_name,
                input_tokens=n_input_tokens,
                output_tokens=n_output_tokens,
                event_log_path=DEFAULT_EVENT_LOG_PATH,
                pricing_csv_path=DEFAULT_PRICING_CSV_PATH,
            )  # Removed timestamp=None, request_id=None as they are default in function
            logger.info("Token usage event logged to output/token_usage_events.jsonl.")
        except Exception as log_exc:
            logger.error(
                f"[Token Logging Error] Failed to log token usage event: {log_exc}"
            )
    except Exception as cost_exception:
        logger.error(f"[Cost Estimation Error] {cost_exception}")


def process_file_common(
    filepath: str,
    output_dir: str,  # Not used directly in this function, but passed down
    config: Dict[str, Any],
    system_prompt_content: str,
    response_field: str,
    encoder: Any,
    token_limit: int,
) -> Tuple[bool, Optional[pd.DataFrame]]:
    """
    Common processing logic for both sequential and concurrent file processing.
    """
    logger.info(f"Processing file: {filepath}")

    def _load_and_validate(current_filepath: str) -> Optional[pd.DataFrame]:
        p_filepath = Path(current_filepath)
        if not p_filepath.exists():
            raise BatchGraderFileNotFoundError(
                f"Input file not found: {current_filepath}"
            )
        df_loaded = load_data(current_filepath)
        if df_loaded is None:
            raise FileFormatError(f"Failed to load data from {current_filepath}")
        logger.info(f"Loaded {len(df_loaded)} rows from {current_filepath}")
        if df_loaded.empty:
            logger.info(f"No data loaded from {current_filepath}. Skipping.")
            return None
        return df_loaded

    def _check_token_limits_wrapper(
        df_to_check: pd.DataFrame,
        sys_prompt: str,
        resp_field: str,
        enc: Any,
        tok_limit: int,
        file_path_str: str,
    ) -> Dict[str, float]:
        # check_token_limits will raise TokenLimitError if limits are exceeded.
        # This includes logging a zero-cost event.
        is_under_limit, token_stats_val = check_token_limits(
            df_to_check,
            sys_prompt,
            resp_field,
            enc,
            tok_limit,
            filepath=file_path_str,
            raise_on_error=True,  # Important: this makes it raise TokenLimitError
        )
        # If no error was raised by check_token_limits, it means it passed.
        # The token logging for *successful pre-check* is handled within check_token_limits itself.
        return token_stats_val

    def _process_with_llm(
        df_to_process: pd.DataFrame,
        sys_prompt: str,
        resp_field: str,
        file_path_str: str,
    ) -> pd.DataFrame:
        llm_client = LLMClient(config=config)  # Pass config to client
        # The original had `if not hasattr(llm_client, "run_batch_job"): return df`
        # This check should ideally not be needed if LLMClient always has this method.
        # Assuming it does for now.
        try:
            return llm_client.run_batch_job(
                df_to_process,
                sys_prompt,
                response_field_name=resp_field,
                base_filename_for_tagging=os.path.basename(file_path_str),
            )
        except Exception as batch_exc:
            raise APIError(
                f"Batch job failed for {file_path_str}: {batch_exc}"
            ) from batch_exc

    try:
        df = _load_and_validate(filepath)
        if df is None:
            return False, None  # No data to process

        token_stats = _check_token_limits_wrapper(
            df,
            system_prompt_content,
            response_field,
            encoder,
            token_limit,
            filepath,  # Pass as positional argument
        )

        if len(df) > MAX_BATCH_SIZE:
            logger.warning(
                f"[WARN] Input file contains {len(df)} rows. Only the first {MAX_BATCH_SIZE} "
                f"will be sent to the API (limit is {MAX_BATCH_SIZE:,} per batch)."
            )
            df = df.iloc[:MAX_BATCH_SIZE].copy()

        force_chunk_count = config.get("force_chunk_count", 0)
        # Corrected logic for use_concurrent
        split_token_limit_config = config.get(
            "split_token_limit", DEFAULT_SPLIT_TOKEN_LIMIT
        )
        use_concurrent = force_chunk_count > 1 or (
            token_stats and token_stats.get("total", 0) > split_token_limit_config
        )

        if use_concurrent:
            # Note: process_file_concurrently needs api_key_prefix, which is derived from LLMClient.
            # Consider how LLMClient is instantiated here if api_key_prefix is strictly needed.
            # For now, assuming it can get it or it's passed via config.
            llm_client_temp = LLMClient(
                config=config
            )  # To get api_key_prefix if needed
            df_with_results = process_file_concurrently(
                filepath=filepath,  # Original filepath for splitting
                config=config,
                system_prompt_content=system_prompt_content,
                response_field=response_field,
                llm_model_name=config.get("openai_model_name", DEFAULT_MODEL),
                api_key_prefix=llm_client_temp.api_key_prefix,
                tiktoken_encoding_func=encoder,
            )
        else:
            df_with_results = _process_with_llm(
                df, system_prompt_content, response_field, filepath
            )

        if df_with_results is not None:
            # Check for 'llm_score' which might indicate errors from run_batch_job
            # This error checking logic might be specific to how run_batch_job signals errors.
            if (
                "llm_score" in df_with_results.columns
            ):  # Assuming llm_score may contain error string
                error_rows = (
                    df_with_results["llm_score"]
                    .astype(str)
                    .str.contains("Error", case=False, na=False)
                )
                if error_rows.any():
                    logger.info(f"Total rows with errors: {error_rows.sum()}")
                    if error_rows.sum() == len(df_with_results):
                        raise DataValidationError(
                            f"All rows failed for {filepath}. Halting further processing."
                        )

            model_name = config.get("openai_model_name", DEFAULT_MODEL)
            # Ensure API key is available
            temp_llm_client = LLMClient(config=config)
            calculate_and_log_token_usage(
                df_with_results,
                system_prompt_content,
                response_field,
                encoder,
                model_name,
                temp_llm_client.api_key,
            )
            return True, df_with_results
        else:
            # This case implies _process_with_llm or process_file_concurrently returned None
            raise DataValidationError(f"No results obtained for {filepath}")

    except (
        FileNotFoundError,
        FileFormatError,
        DataValidationError,
        TokenLimitError,  # This will be raised by _check_token_limits_wrapper if exceeded
        APIError,
    ):
        raise  # Re-raise known, handled exceptions
    except Exception as e:
        # Catch-all for other unexpected errors during the process
        logger.error(
            f"Unexpected error in process_file_common for {filepath}: {e}",
            exc_info=True,
        )
        raise FileProcessingError(
            f"An unexpected error occurred while processing {filepath}: {e}"
        ) from e


def process_file_wrapper(
    filepath: str,
    output_dir: str,
    config: Dict[str, Any],
    system_prompt_content: str,
    response_field: str,
    encoder: Any,
    token_limit: int,
    is_reprocessing_run: bool = False,
) -> bool:
    """
    Process a single file, handling token limits with potential splitting and saving results.
    """
    try:
        # Initial attempt to process the file as a whole
        output_path = prepare_output_path(
            filepath, output_dir, config, is_reprocessing_run
        )
        success, df_with_results = process_file_common(
            filepath,
            output_dir,  # Pass output_dir, though process_file_common might not use it directly
            config,
            system_prompt_content,
            response_field,
            encoder,
            token_limit,
        )

        if success and df_with_results is not None:
            # Load original to check for custom_id
            original_df = load_data(filepath)
            if (
                original_df is not None
                and "custom_id" not in original_df.columns
                and "custom_id" in df_with_results.columns
            ):
                df_with_results = df_with_results.drop(
                    columns=["custom_id"], errors="ignore"
                )
                logger.debug(
                    f"Dropped generated 'custom_id' column from results for {filepath}"
                )
            save_data(df_with_results, output_path)
            logger.info(  # Changed from logger.success to logger.info for broader compatibility
                f"Processed {filepath}. Results saved to {output_path}"
            )
            logger.info(f"Total rows successfully processed: {len(df_with_results)}")
            return True
        # If process_file_common returned False but didn't raise TokenLimitError, it's some other failure
        elif not success:
            logger.error(
                f"Processing failed for {filepath} but no specific error was re-raised to wrapper."
            )
            return False

    except TokenLimitError as e:
        logger.debug(f"[TokenLimitError Caught in Wrapper] {str(e)}")
        logger.info(
            "The file exceeds the token limit. Would you like to automatically split it "
            "into smaller chunks and process them? (y/n)"
        )
        logger.info("(Will proceed automatically after 30 seconds if no response)")

        import threading  # Keep import local if only used here

        user_response_container = {"response": None}  # Use a mutable container

        def get_input_with_timeout():
            from contextlib import suppress

            with suppress(EOFError):  # Handle EOF if stdin is closed (e.g. in tests)
                try:
                    user_response_container["response"] = input()
                except RuntimeError:  # Can happen if stdin is not a tty
                    logger.warning("Could not read from stdin for user prompt.")
                    user_response_container["response"] = "y"  # Default to yes

        input_thread = threading.Thread(target=get_input_with_timeout)
        input_thread.daemon = True
        input_thread.start()
        input_thread.join(timeout=30)

        user_choice = user_response_container["response"]
        if user_choice is None:  # Timeout
            logger.info(
                "No response received within 30 seconds. Proceeding with automatic chunking."
            )
            proceed_with_splitting = True
        elif user_choice.lower() in ["n", "no"]:
            logger.info("User chose not to split the file. Aborting.")
            return False
        else:  # Default to yes for "y", "yes", empty string, or any other input
            logger.info("Proceeding with automatic chunking.")
            proceed_with_splitting = True

        if not proceed_with_splitting:
            return False

        logger.info(f"Splitting file {filepath} into chunks...")
        # split_token_limit is fetched by _generate_chunk_job_objects from config
        model_name = config.get("openai_model_name", DEFAULT_MODEL)
        api_key_prefix = None
        try:
            llm_client = LLMClient(config=config)
            api_key_prefix = llm_client.api_key_prefix
        except Exception as client_err:
            logger.warning(
                f"Could not initialize LLMClient for API key prefix: {client_err}"
            )

        # Process the chunks concurrently (process_file_concurrently handles job generation)
        df_result_from_chunks = process_file_concurrently(
            filepath=filepath,  # Original filepath for splitting
            config=config,
            system_prompt_content=system_prompt_content,
            response_field=response_field,
            llm_model_name=model_name,
            api_key_prefix=api_key_prefix,
            tiktoken_encoding_func=encoder,
        )

        if df_result_from_chunks is not None:
            chunked_output_suffix = "_chunked_results"
            p_filepath = Path(filepath)
            base_name = p_filepath.stem
            if is_reprocessing_run:
                base_name += "_reprocessed"

            config.copy()  # this is ultra scuff

            output_dir_path = Path(output_dir)
            output_filename_chunked = (
                f"{Path(filepath).stem}{chunked_output_suffix}{Path(filepath).suffix}"
            )
            if is_reprocessing_run:
                output_filename_chunked = f"{Path(filepath).stem}_reprocessed{chunked_output_suffix}{Path(filepath).suffix}"

            final_output_path_chunked = output_dir_path / output_filename_chunked

            # Handle if this path also exists by timestamping (like prepare_output_path does)
            if final_output_path_chunked.exists():
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                final_output_path_chunked = (
                    output_dir_path
                    / f"{Path(output_filename_chunked).stem}_{timestamp}{Path(output_filename_chunked).suffix}"
                )

            save_data(df_result_from_chunks, str(final_output_path_chunked))
            logger.info(
                f"Processed {filepath} in chunks. Results saved to {final_output_path_chunked}"
            )
            logger.info(
                f"Total rows successfully processed from chunks: {len(df_result_from_chunks)}"
            )
            return True
        else:
            logger.error(f"Failed to process chunks for {filepath}.")
            return False

    except (
        FileNotFoundError,
        FileFormatError,
        DataValidationError,
        APIError,
        OutputDirectoryError,
        FilePermissionError,
        FileProcessingError,  # From process_file_common if it raises a general one
    ) as e:
        logger.error(
            f"[HANDLED ERROR in Wrapper] {str(e)} for {filepath}", exc_info=False
        )  # exc_info=False to reduce noise for handled errors
        return False
    except Exception as e:
        logger.error(
            f"[CRITICAL ERROR] Unexpected error in process_file_wrapper for {filepath}: {e}",
            exc_info=True,
        )
        return False


def _generate_chunk_job_objects(
    original_filepath: str,
    system_prompt_content: str,
    config: dict,
    tiktoken_encoding_func: Any,
    response_field: str,
    llm_model_name: Optional[str],
    api_key_prefix: Optional[str],
) -> list[BatchJob]:
    """Splits the input file and creates BatchJob objects for each chunk."""
    splitter_config = config.get("input_splitter_options", {})
    default_split_limit = config.get("split_token_limit", DEFAULT_SPLIT_TOKEN_LIMIT)
    max_tokens_per_chunk = splitter_config.get(
        "max_tokens_per_chunk", default_split_limit
    )
    max_rows_per_chunk = splitter_config.get(
        "max_rows_per_chunk", config.get("split_row_limit")
    )  # Can be None
    force_chunk_count_val = splitter_config.get("force_chunk_count", None)

    logger.debug(f"Generating chunk job objects for: {original_filepath}")
    logger.debug(
        f"Using splitter config: max_tokens={max_tokens_per_chunk}, max_rows={max_rows_per_chunk}, force_chunks={force_chunk_count_val}"
    )

    if not (
        callable(tiktoken_encoding_func)
        or (
            hasattr(tiktoken_encoding_func, "encode")
            and callable(tiktoken_encoding_func.encode)
        )
    ):
        logger.error(
            f"Invalid tiktoken_encoding_func passed. Type: {type(tiktoken_encoding_func)}. Attempting to load default."
        )
        try:
            import tiktoken  # Local import for safety

            tiktoken_encoding_func = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.critical(
                f"Failed to load default tiktoken encoder: {e}. Chunking will likely fail.",
                exc_info=True,
            )
            return []

    def _count_row_tokens_for_splitter(row: pd.Series) -> int:
        # This function must match the token counting logic used for pre-checks if consistency is paramount
        # For splitting, we typically count tokens for the actual content that will be sent.
        # Assuming system_prompt is applied per request, not per row within a batch sent to LLM.
        # If system_prompt is part of *each* item in a batch, it should be included here.
        # The provided create_token_counter combines system_prompt + response_field.
        # For splitting, we usually care about input tokens per row.
        # Let's make this consistent with how create_token_counter_util does it for *input parts*.

        # Simplified: count tokens for all relevant fields in the row that form the input.
        # The `response_field` is typically for the *output*, not for splitting input.
        # The `system_prompt_content` is usually static for the whole batch.
        # So, the tokens per row for splitting should be based on the row's actual content.

        # This logic should be robust. If `system_prompt_content` is a column name:
        current_system_prompt = str(
            row.get(system_prompt_content, system_prompt_content)
            if isinstance(system_prompt_content, str) and system_prompt_content in row
            else system_prompt_content
        )

        row_content_parts = [
            str(value)
            for col, value in row.items()
            if col != response_field and pd.notna(value)
        ]
        row_text = " ".join(row_content_parts)

        full_content_for_tokenization = current_system_prompt + "\n" + row_text

        if callable(tiktoken_encoding_func):
            return len(tiktoken_encoding_func(full_content_for_tokenization))
        elif hasattr(tiktoken_encoding_func, "encode"):
            return len(tiktoken_encoding_func.encode(full_content_for_tokenization))
        # Fallback, though previous check should catch invalid encoder
        logger.error("Splitter's token counting failed due to invalid encoder.")
        return 1_000_000  # Penalize to avoid infinite loops if encoder fails

    try:
        chunk_file_paths, _ = split_file_by_token_limit(
            input_path=original_filepath,
            token_limit=max_tokens_per_chunk,
            # Use the specifically defined counter for splitting
            count_tokens_fn=_count_row_tokens_for_splitter,
            # Passed to splitter, its role there needs to be clear
            response_field=response_field,
            row_limit=max_rows_per_chunk,
            force_chunk_count=force_chunk_count_val,
            logger_override=logger,
        )
    except Exception as e:
        logger.error(
            f"Error during file splitting for {original_filepath}: {e}", exc_info=True
        )
        return []

    if not chunk_file_paths:
        logger.warning(
            f"No chunk files generated by input_splitter for {original_filepath}."
        )
        return []

    jobs = []
    for i, chunk_path_str in enumerate(chunk_file_paths):
        chunk_path = Path(chunk_path_str)
        try:
            logger.debug(f"Loading chunk file: {chunk_path}")
            chunk_df = load_data(str(chunk_path))
            if chunk_df is None or chunk_df.empty:
                logger.warning(f"Skipping empty or unreadable chunk file: {chunk_path}")
                continue

            if "custom_id" in chunk_df.columns and "id" in chunk_df.columns:
                logger.info(
                    f"Both 'id' and 'custom_id' columns found in chunk: {chunk_path.name}. Using 'custom_id'."
                )
            elif "id" in chunk_df.columns:
                chunk_df = chunk_df.rename(columns={"id": "custom_id"})
                logger.info(f"Renamed 'id' to 'custom_id' for chunk: {chunk_path.name}")

            if "custom_id" in chunk_df.columns:
                if not pd.api.types.is_string_dtype(chunk_df["custom_id"]):
                    logger.debug(
                        f"Casting 'custom_id' to string for chunk: {chunk_path.name} (type: {chunk_df['custom_id'].dtype})"
                    )
                    chunk_df["custom_id"] = chunk_df["custom_id"].astype(str)
            else:
                logger.warning(
                    f"Neither 'custom_id' nor 'id' column found in chunk: {chunk_path.name}."
                )

            job = BatchJob(
                chunk_id_str=chunk_path.stem,
                chunk_df=chunk_df,
                system_prompt=system_prompt_content,
                response_field=response_field,
                original_filepath=original_filepath,
                chunk_file_path=str(chunk_path),
                llm_model=llm_model_name,
                api_key_prefix=api_key_prefix,
                status="pending",
            )
            jobs.append(job)
        except Exception as e:
            logger.error(
                f"Error processing chunk file {chunk_path} into BatchJob: {e}",
                exc_info=True,
            )
            # Create a placeholder failed job
            failed_job = BatchJob(
                chunk_id_str=(chunk_path.stem if chunk_path else f"failed_chunk_{i}"),
                chunk_df=None,
                system_prompt=system_prompt_content,
                response_field=response_field,
                original_filepath=original_filepath,
                chunk_file_path=str(chunk_path) if chunk_path else "N/A",
                llm_model=llm_model_name,
                api_key_prefix=api_key_prefix,
                status="error",
                error_message=f"Failed to load/prepare BatchJob from chunk: {e}",
                error_details=traceback.format_exc(),
            )
            jobs.append(failed_job)

    logger.info(f"Generated {len(jobs)} BatchJob objects from {original_filepath}.")
    return jobs


def _execute_single_batch_job_task(
    batch_job: BatchJob,
    llm_client_instance: Optional[LLMClient] = None,  # Renamed for clarity
    response_field_name: str = "response",
    config_for_client: Optional[Dict[str, Any]] = None,  # Renamed for clarity
) -> BatchJob:
    """Worker function to process a single BatchJob."""
    if batch_job.chunk_df is None or batch_job.chunk_df.empty:
        if batch_job.status != "error":  # Avoid overriding existing error status
            batch_job.status = "error"
            batch_job.error_message = (
                "Chunk DataFrame is None or empty at task execution."
            )
        logger.error(
            f"[{batch_job.chunk_id_str}] Skipping task: {batch_job.error_message}"
        )
        return batch_job

    current_llm_client = llm_client_instance
    if current_llm_client is None:
        try:
            current_llm_client = LLMClient(config=config_for_client)
            logger.debug(f"[{batch_job.chunk_id_str}] Created new LLMClient for thread")
        except Exception as e:
            batch_job.status = "error"
            batch_job.error_message = f"Failed to create LLMClient: {str(e)}"
            batch_job.error_details = traceback.format_exc()
            logger.error(f"[{batch_job.chunk_id_str}] {batch_job.error_message}")
            return batch_job

    try:
        batch_job.status = "running"
        logger.info(f"[{batch_job.chunk_id_str}] Task starting.")

        if "custom_id" not in batch_job.chunk_df.columns:
            logger.warning(
                f"[{batch_job.chunk_id_str}] 'custom_id' column missing at task start."
            )

        api_result = current_llm_client.run_batch_job(
            batch_job.chunk_df,
            batch_job.system_prompt,
            response_field_name=response_field_name,
            base_filename_for_tagging=batch_job.chunk_id_str,
        )

        if isinstance(api_result, pd.DataFrame):
            batch_job.result_data = api_result
            batch_job.status = "completed"
            logger.debug(
                f"[{batch_job.chunk_id_str}] Task completed, result: DataFrame {len(api_result)} rows."
            )
        elif isinstance(api_result, dict) and (
            "error" in api_result or "custom_id_of_failed_item" in api_result
        ):
            batch_job.status = "failed"  # API specific failure for a batch
            batch_job.error_message = api_result.get(
                "error_message", api_result.get("error", str(api_result))
            )
            batch_job.error_details = api_result
            logger.warning(
                f"[{batch_job.chunk_id_str}] Task API failure: {batch_job.error_message}"
            )
        else:
            batch_job.status = "error"  # Unexpected result type
            batch_job.error_message = f"Unexpected API result type: {type(api_result)}"
            batch_job.error_details = str(api_result)
            logger.error(
                f"[{batch_job.chunk_id_str}] Task error: {batch_job.error_message}"
            )

    except Exception as exc:
        logger.error(
            f"[{batch_job.chunk_id_str}] Exception in task: {exc}",
            exc_info=True,
        )
        batch_job.status = "failed"  # General task failure
        batch_job.error_message = str(exc)
        batch_job.error_details = traceback.format_exc()

    logger.info(
        f"[{batch_job.chunk_id_str}] Task finished with status: {batch_job.status}"
    )
    return batch_job


def _pfc_submit_jobs(
    jobs_to_submit: List[BatchJob],
    response_field_name: str,
    max_workers_config: int,
    config_for_llmclient: Optional[Dict[str, Any]] = None,
) -> Dict[Any, BatchJob]:
    """Submits BatchJob objects to a ThreadPoolExecutor."""
    future_to_job_map = {}
    # Intentionally create LLMClient inside _execute_single_batch_job_task for thread safety
    with ThreadPoolExecutor(max_workers=max_workers_config) as executor:
        for job in jobs_to_submit:
            future = executor.submit(
                _execute_single_batch_job_task,
                job,
                None,  # Pass None for llm_client_instance, task will create it
                response_field_name,
                config_for_llmclient,  # Pass config for client instantiation
            )
            future_to_job_map[future] = job
    logger.info(f"Submitted {len(future_to_job_map)} chunk jobs to executor.")
    return future_to_job_map


def _handle_step_error(
    step_name: str, e: Exception, filepath: Optional[str] = None
) -> None:
    file_msg = f" for {Path(filepath).name}" if filepath else ""
    logger.error(f"{step_name}{file_msg} failed: {e}", exc_info=True)
    # This function returns None implicitly, its purpose is logging.


def _pfc_process_completed_future(
    future: Any,
    future_to_job_map: Dict[Any, BatchJob],
    completed_jobs_list: List[BatchJob],
    live_display: Optional[Live],
    rich_table: RichJobTable,
    all_jobs_list: List[BatchJob],
    halt_on_failure_flag: bool,
    original_filepath_str: str,  # Renamed for clarity
    llm_output_col_name: str,  # Renamed for clarity
) -> bool:
    """Processes a single completed future. Returns True if halt is signaled."""

    def _create_error_dataframe(
        job_obj: BatchJob, error_custom_id_val=None
    ) -> pd.DataFrame:
        """Helper to create a DataFrame representing an error for a job."""
        if error_custom_id_val is None:
            # Try to get custom_id from chunk_df if available, else use chunk_id_str
            if (
                job_obj.chunk_df is not None
                and "custom_id" in job_obj.chunk_df.columns
                and not job_obj.chunk_df["custom_id"].empty
            ):
                # Potentially multiple IDs if chunk_df exists. Use first or a generic marker.
                # For simplicity, let's use chunk_id_str if actual IDs are problematic here.
                error_custom_id_val = (
                    job_obj.chunk_id_str
                )  # Or handle multiple IDs from chunk_df
            else:
                error_custom_id_val = job_obj.chunk_id_str

        error_data = {
            "custom_id": error_custom_id_val,
            llm_output_col_name: f"ERROR: {job_obj.error_message or 'Unknown processing error'}",
            "error_type": (
                str(type(job_obj.error_details).__name__)
                if job_obj.error_details
                else (job_obj.status.capitalize() + "Error")
            ),
            "original_file": original_filepath_str,
            "chunk_id": job_obj.chunk_id_str,
            "error_details": (
                str(job_obj.error_details) if job_obj.error_details else "No details"
            ),
        }
        return pd.DataFrame([error_data])

    job_from_map = future_to_job_map[future]
    logger.info(
        f"Future completed for chunk {job_from_map.chunk_id_str}. Processing..."
    )

    try:
        # Retrieve the updated BatchJob object from the future's result
        processed_job_from_task: BatchJob = future.result()

        # Sync status from processed_job_from_task back to job_from_map (which is also in all_jobs_list)
        job_from_map.status = processed_job_from_task.status
        job_from_map.error_message = processed_job_from_task.error_message
        job_from_map.error_details = processed_job_from_task.error_details
        job_from_map.result_data = processed_job_from_task.result_data

        logger.info(
            f"[{job_from_map.chunk_id_str}] Status from task: {job_from_map.status}"
        )

        if job_from_map.status == "completed":
            if isinstance(job_from_map.result_data, pd.DataFrame):
                completed_jobs_list.append(job_from_map)
                logger.info(  # Changed from success for consistency
                    f"Chunk {job_from_map.chunk_id_str} completed successfully."
                )
            else:  # Should not happen if task logic is correct for "completed"
                job_from_map.status = "error"
                job_from_map.error_message = f"Completed but result_data is not DataFrame (type: {type(job_from_map.result_data)})"
                logger.error(
                    f"Chunk {job_from_map.chunk_id_str}: {job_from_map.error_message}"
                )
                job_from_map.result_data = _create_error_dataframe(job_from_map)
                completed_jobs_list.append(
                    job_from_map
                )  # Still add, it's "completed" in terms of processing attempt

        elif job_from_map.status in ("failed", "error"):
            logger.error(
                f"Chunk {job_from_map.chunk_id_str} {job_from_map.status}. Error: {job_from_map.error_message}"
            )
            # Ensure it has an error dataframe and is added to completed_jobs_list for aggregation
            if (
                job_from_map not in completed_jobs_list
            ):  # Avoid double-adding if already processed
                if job_from_map.result_data is None or not isinstance(
                    job_from_map.result_data, pd.DataFrame
                ):
                    job_from_map.result_data = _create_error_dataframe(job_from_map)
                completed_jobs_list.append(job_from_map)

            if halt_on_failure_flag:
                logger.error(
                    f"[HALT] Failure in {job_from_map.chunk_id_str}. Halting {Path(original_filepath_str).name}."
                )
                if live_display:
                    live_display.update(rich_table.build_table(all_jobs_list))
                return True  # Signal halt

    except (
        Exception
    ) as e:  # Exception during future.result() or subsequent processing here
        job_from_map.status = "error"
        job_from_map.error_message = (
            f"Exception processing future for {job_from_map.chunk_id_str}: {e}"
        )
        job_from_map.error_details = traceback.format_exc()
        logger.error(f"[{job_from_map.chunk_id_str}] Exception: {e}", exc_info=True)

        if job_from_map not in completed_jobs_list:
            if job_from_map.result_data is None or not isinstance(
                job_from_map.result_data, pd.DataFrame
            ):
                job_from_map.result_data = _create_error_dataframe(job_from_map)
            completed_jobs_list.append(job_from_map)

        if halt_on_failure_flag:
            logger.error(
                f"[HALT] Future error. Halting {Path(original_filepath_str).name}."
            )
            if live_display:
                live_display.update(rich_table.build_table(all_jobs_list))
            return True  # Signal halt

    if live_display:
        live_display.update(rich_table.build_table(all_jobs_list))

    # Final check on status for halt, in case it was set to error/failed within try
    return job_from_map.status in {"failed", "error"} and halt_on_failure_flag


def _pfc_monitor_and_process_futures(
    future_to_job_map: Dict[Any, BatchJob],
    completed_jobs_list: List[BatchJob],
    all_jobs_list: List[BatchJob],  # This is the list of original job objects
    rich_table: RichJobTable,
    halt_on_failure_flag: bool,
    original_filepath_str: str,
    llm_output_col_name: str,
) -> bool:  # Returns True if halted, False otherwise
    """Monitors job futures, processes them, and handles halting."""
    halt_signaled = False
    try:
        with Live(
            rich_table.build_table(all_jobs_list),  # Initial table
            console=rich_table.console,
            refresh_per_second=4,
            vertical_overflow="visible",
        ) as live:
            for future in as_completed(future_to_job_map.keys()):
                if _pfc_process_completed_future(
                    future=future,
                    future_to_job_map=future_to_job_map,
                    completed_jobs_list=completed_jobs_list,
                    live_display=live,  # Pass the Live instance
                    rich_table=rich_table,
                    all_jobs_list=all_jobs_list,  # Pass the main list for table updates
                    halt_on_failure_flag=halt_on_failure_flag,
                    original_filepath_str=original_filepath_str,
                    llm_output_col_name=llm_output_col_name,
                ):
                    halt_signaled = True
                    logger.error(
                        f"Halt signal received for {original_filepath_str}. Cancelling remaining."
                    )
                    # Cancel remaining futures
                    for fut_to_cancel, job_to_cancel in future_to_job_map.items():
                        if not fut_to_cancel.done() and fut_to_cancel.cancel():
                            job_to_cancel.status = (
                                "cancelled"  # Update status on the original job object
                            )
                            logger.debug(f"Cancelled job {job_to_cancel.chunk_id_str}.")
                    break  # Exit as_completed loop
            live.update(rich_table.build_table(all_jobs_list))  # Final update
    except Exception as e:
        _handle_step_error("Job monitoring", e, original_filepath_str)
        return True  # Treat unhandled exception in monitoring as a reason to halt/indicate failure
    return halt_signaled


def _pfc_aggregate_and_cleanup(
    processed_jobs_list: List[
        BatchJob
    ],  # Renamed as it contains processed jobs (completed/failed/error)
    original_filepath_str: str,
    # response_field_name: str, # Not directly used here for aggregation logic itself
) -> Optional[pd.DataFrame]:
    """Aggregates results from BatchJob objects and cleans up chunked files."""
    all_results_dfs = []
    total_successful_rows = 0

    for job in processed_jobs_list:
        if isinstance(job.result_data, pd.DataFrame) and not job.result_data.empty:
            all_results_dfs.append(job.result_data)
            if job.status == "completed":  # Only count rows from truly successful jobs
                total_successful_rows += len(job.result_data)
        elif job.status == "completed" and (
            job.result_data is None or isinstance(job.result_data, pd.DataFrame)
        ):
            logger.warning(
                f"Job {job.chunk_id_str} status 'completed' but no valid result_data. Skipping."
            )
            # For failed/error jobs, their result_data (if it's an error DataFrame) will be included

    original_file_path_obj = Path(original_filepath_str)
    chunked_dir_to_clean = (
        original_file_path_obj.parent / f"_chunked_{original_file_path_obj.stem}"
    )

    if not all_results_dfs:
        logger.error(
            f"No valid results to aggregate for {original_file_path_obj.name}."
        )
        # Attempt cleanup even if no results
        cleanup_chunked_dir(chunked_dir_to_clean)
        return None

    try:
        combined_df = pd.concat(all_results_dfs, ignore_index=True)
        logger.info(  # Changed from success
            f"Aggregated for {original_file_path_obj.name}. Combined: {len(combined_df)} rows ({total_successful_rows} from successful chunks)."
        )

        # ID column handling: if 'custom_id' exists, ensure 'id' also does, copying if necessary.
        # This assumes 'custom_id' is the primary one from batch processing.
        if "custom_id" in combined_df.columns and "id" not in combined_df.columns:
            combined_df["id"] = combined_df["custom_id"]
            logger.debug(
                f"Copied 'custom_id' to 'id' for {original_file_path_obj.name}."
            )

        cleanup_chunked_dir(chunked_dir_to_clean)
        return combined_df

    except Exception as e:
        logger.error(
            f"Failed to concatenate results for {original_file_path_obj.name}: {e}",
            exc_info=True,
        )
        cleanup_chunked_dir(chunked_dir_to_clean)  # Attempt cleanup on error
        return None


def process_file_concurrently(
    filepath: str,
    config: Dict[str, Any],
    system_prompt_content: str,
    response_field: str,
    llm_model_name: str,  # Already in config, but passed explicitly
    # Already in config (via LLMClient), but passed
    api_key_prefix: Optional[str],
    # Already in config (via LLMClient/encoder), but passed
    tiktoken_encoding_func: Any,
) -> Optional[pd.DataFrame]:
    """
    Process a file by splitting into chunks and processing them concurrently.
    """
    logger.info(f"Starting concurrent processing for: {filepath}")

    # Step 1: Generate jobs
    try:
        jobs: List[BatchJob] = _generate_chunk_job_objects(
            original_filepath=filepath,
            config=config,  # Contains splitter_options, split_token_limit etc.
            system_prompt_content=system_prompt_content,
            response_field=response_field,
            llm_model_name=llm_model_name,
            api_key_prefix=api_key_prefix,
            tiktoken_encoding_func=tiktoken_encoding_func,
        )
    except (
        Exception
    ) as e:  # Catch-all for safety, though _generate should handle its own
        _handle_step_error("Job generation", e, filepath)
        return None

    if not jobs:
        logger.warning(f"No BatchJob objects generated for {filepath}. Cannot proceed.")
        # Attempt to cleanup if a chunked directory was made by splitter but no jobs resulted
        original_file_path_obj = Path(filepath)
        chunked_dir_potential = (
            original_file_path_obj.parent / f"_chunked_{original_file_path_obj.stem}"
        )
        if chunked_dir_potential.exists():
            cleanup_chunked_dir(chunked_dir_potential)
        return None

    halt_on_failure = config.get("halt_on_chunk_failure", True)
    max_workers_val = (
        config.get("max_simultaneous_batches") or config.get("max_workers") or 2
    )  # Default to 2 if not found
    if not (isinstance(max_workers_val, int) and max_workers_val > 0):
        logger.warning(f"Invalid max_workers '{max_workers_val}', defaulting to 2.")
        max_workers_val = 2

    # Step 2: Submit jobs
    try:
        future_to_job_map: Dict[Any, BatchJob] = _pfc_submit_jobs(
            jobs,
            response_field,
            max_workers_val,
            config,  # Pass main config for LLMClient
        )
    except Exception as e:
        _handle_step_error("Job submission", e, filepath)
        return None  # Cannot proceed if submission fails

    processed_jobs_collector: List[BatchJob] = (
        []
    )  # Collects all jobs after they are processed (completed, failed, error)
    rich_table = RichJobTable()  # For display

    # Determine the actual column name where LLM output is expected/stored.
    # This might be `response_field` or a more specific one from config.
    llm_actual_output_column = config.get("llm_output_column_name", response_field)

    # Step 3: Monitor and process futures
    halted = _pfc_monitor_and_process_futures(
        future_to_job_map=future_to_job_map,
        # This list will be populated by the monitor
        completed_jobs_list=processed_jobs_collector,
        all_jobs_list=jobs,  # Pass the original list of jobs for display
        rich_table=rich_table,
        halt_on_failure_flag=halt_on_failure,
        original_filepath_str=filepath,
        llm_output_col_name=llm_actual_output_column,
    )
    # `halted` being None would indicate an error within _pfc_monitor_and_process_futures itself.
    if halted is None:  # Critical error in monitoring
        logger.error(f"Critical error in job monitoring for {filepath}. Aborting.")
        return None

    # Step 4: Aggregate results and cleanup
    # This step runs regardless of `halted` status to process any jobs that did complete.
    logger.info(
        f"Futures processing finished for {filepath} (Halted={halted}). Aggregating results..."
    )
    final_df = _pfc_aggregate_and_cleanup(
        processed_jobs_list=processed_jobs_collector,  # Use the populated list
        original_filepath_str=filepath,
        # response_field_name=llm_actual_output_column # Pass the correct output column name
    )

    if final_df is not None:
        logger.info(  # Changed from success
            f"Concurrent processing of {filepath} yielded aggregated DataFrame with {len(final_df)} rows."
        )
    else:
        logger.error(
            f"Concurrent processing of {filepath} did not produce an aggregated DataFrame."
        )
    return final_df


def cleanup_chunked_dir(chunked_dir_path: Path) -> None:
    """Remove the given directory of chunked files if it exists."""
    if not chunked_dir_path.is_dir():
        logger.debug(
            "No chunked directory found at %s; skipping cleanup.", chunked_dir_path
        )
        return

    try:
        # prune_chunked_dir (from file_utils) should handle actual deletion
        prune_chunked_dir(
            str(chunked_dir_path)
        )  # Assumes prune_chunked_dir deletes files and the dir
        logger.info("Successfully pruned chunked directory %s", chunked_dir_path)
    except Exception as exc:
        logger.error(
            "Error pruning chunked directory %s: %s",
            chunked_dir_path,
            exc,
            exc_info=True,
        )


# Public API for file_processor module
__all__ = [
    "check_token_limits",
    "prepare_output_path",
    "calculate_and_log_token_usage",
    "process_file_common",  # Exposes common logic if needed externally
    "process_file_wrapper",  # Main entry point for single file processing
    "process_file_concurrently",  # Exposes concurrent logic
    "cleanup_chunked_dir",  # Utility
    "ProcessingStats",  # Dataclass
]
