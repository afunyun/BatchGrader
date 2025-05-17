from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import datetime
import logging
import sys
import tempfile # Added for temporary reprocessing file
import uuid # Added for unique temp file names

import pandas as pd

from src.config_loader import load_config
from src.constants import (DEFAULT_GLOBAL_TOKEN_LIMIT, DEFAULT_MODEL,
                           DEFAULT_RESPONSE_FIELD, DEFAULT_SPLIT_TOKEN_LIMIT,
                           MAX_BATCH_SIZE, DEFAULT_EVENT_LOG_PATH,
                           DEFAULT_TOKEN_USAGE_LOG_PATH)
from src.data_loader import load_data, save_data
from src.file_processor import check_token_limits, prepare_output_path, process_file_wrapper
from src.input_splitter import split_file_by_token_limit
from src.llm_client import LLMClient
from src.log_utils import prune_logs_if_needed
from src.prompt_utils import load_system_prompt
from src.token_tracker import get_token_usage_for_day, get_token_usage_summary, log_token_usage_event, update_token_log
from src.utils import get_encoder

logger = logging.getLogger(__name__)


def get_log_dirs(args) -> Tuple[Path, Path]:
    """
    Get log and archive directories, either from args or using defaults from constants.
    
    Args:
        args: Command line arguments which may include log_dir
        
    Returns:
        Tuple of (log_dir, archive_dir) as Path objects
    """
    from src.constants import DEFAULT_LOG_DIR, DEFAULT_ARCHIVE_DIR

    if hasattr(args, 'log_dir') and args.log_dir:
        log_dir = Path(args.log_dir).resolve()
        archive_dir = log_dir / 'archive'
        logger.info(f"Log directory from CLI: {log_dir}")
        logger.info(f"Archive directory: {archive_dir}")
    else:
        log_dir = DEFAULT_LOG_DIR
        archive_dir = DEFAULT_ARCHIVE_DIR

    return log_dir, archive_dir


def process_file(filepath_str: str, output_dir_str: str, config: Dict[str, Any], args: Any) -> bool:
    """
    Processes a single file using the new file_processor.process_file_wrapper.
    This function now acts as a simpler interface to the core processing logic.
    
    Args:
        filepath_str: Path to the input file as a string
        output_dir_str: Output directory path as a string
        config: Configuration dictionary containing processing parameters
        
    Returns:
        bool: True if processing was successful, False otherwise
    """
    filepath = Path(filepath_str)
    output_dir = Path(output_dir_str)
    logger.info(f"Initiating processing for: {filepath.name} -> {output_dir}")

    try:
        return _process_file_with_config(config, filepath, output_dir, args)
    except FileNotFoundError:
        logger.error(
            f"[ERROR] Input file not found: {filepath}. Please verify the file exists and you have read permissions."
        )
        return False
    except ValueError as ve:
        logger.error(
            f"[ERROR] Configuration or input error for {filepath.name}: {ve}. Current config: {config}"
        )
        return False
    except Exception as e:
        logger.error(
            f"[CRITICAL ERROR] Unexpected error processing {filepath.name}: {e}",
            exc_info=True,
            extra={
                "file_info": {
                    "name": filepath.name,
                    "size": filepath.stat().st_size if filepath.exists() else None,
                    "last_modified": datetime.datetime.fromtimestamp(
                        filepath.stat().st_mtime).isoformat()
                    if filepath.exists() else None
                },
                "config": config,
                "error_type": type(e).__name__
            }
        )
        return False


def _process_file_with_config(
    config: Dict[str, Any], 
    filepath: Path, 
    output_dir: Path,
    args: Any # For is_reprocessing_run flag
) -> bool:
    """Process a file with the given configuration.
    
    Args:
        config: Configuration dictionary
        filepath: Path to the input file
        output_dir: Output directory path
        
    Returns:
        bool: True if processing was successful, False otherwise
    """
    system_prompt_content = load_system_prompt(config)
    response_field = config.get('response_field_name', DEFAULT_RESPONSE_FIELD)

    # Get encoder using centralized utility
    model_name = config.get('openai_model_name', DEFAULT_MODEL)
    encoder = get_encoder(model_name)
    if encoder is None:
        logger.error(
            f"Failed to get encoder for model {model_name}. Token counting and splitting might be affected."
        )

    # Use global token limit from constants, overridden by config if present
    token_limit = config.get('global_token_limit', DEFAULT_GLOBAL_TOKEN_LIMIT)
    logger.debug(
        f"Processing configuration: model={model_name}, token_limit={token_limit}, response_field={response_field}"
    )

    # Delegate to the centralized file processing wrapper
    success = process_file_wrapper(
        filepath=str(filepath),
        output_dir=str(output_dir),
        config=config,
        system_prompt_content=system_prompt_content,
        response_field=response_field,
        encoder=encoder,
        token_limit=token_limit,
        is_reprocessing_run=args.is_reprocessing_run
    )

    if success:
        logger.info(f"Successfully processed {filepath.name}.")
    else:
        logger.info(
            f"Failed to process {filepath.name}. See logs for details. Config state: model={model_name}, token_limit={token_limit}, response_field={response_field}"
        )
    return success


def print_token_cost_stats(config=None):
    'Prints token usage statistics for the current day.'
    try:
        # Pass config to LLMClient instead of having it load config itself
        llm_client = LLMClient(config=config)
        api_key = llm_client.api_key
        if daily_usage := get_token_usage_for_day(api_key, log_path=DEFAULT_TOKEN_USAGE_LOG_PATH):
            logger.info("Token Usage Stats (Today):")
            # Ensure daily_usage is treated as an int if it's just the token count
            if isinstance(daily_usage, int):
                logger.info(f"  Total Tokens: {daily_usage}")
            elif isinstance(daily_usage, dict):
                logger.info(
                    f"  Total Tokens: {daily_usage.get('total_tokens', daily_usage.get('tokens_submitted', 'N/A'))}"
                )
                if 'estimated_cost' in daily_usage:
                    logger.info(
                        f"  Estimated Cost: ${daily_usage['estimated_cost']:.4f}"
                    )
            else:
                logger.info(f"  Usage data: {daily_usage}")
        else:
            logger.info(
                "No token usage recorded for today or failed to retrieve stats."
            )
    except Exception as e:
        logger.error(f"Could not retrieve daily token stats: {e}")


def print_token_cost_summary(summary_file_path: Optional[str] = None,
                             log_dir: Optional[Path] = None,
                             config=None):
    """Prints a summary of token usage from the summary file."""
    try:
        llm_client = LLMClient(config=config)
        api_key = llm_client.api_key

        # Use provided log_dir or get from constants
        if log_dir is None:
            from src.constants import DEFAULT_LOG_DIR
            log_dir = DEFAULT_LOG_DIR

        # Default path if not provided
        default_summary_path = log_dir / f"token_usage_summary_{api_key[:8]}.json"
        actual_summary_path = Path(
            summary_file_path) if summary_file_path else default_summary_path

        if actual_summary_path.exists():
            if summary := get_token_usage_summary(event_log_path=actual_summary_path):
                logger.info("Overall Token Usage Summary (from file):")
                logger.info(
                    f"  Total Tokens (Overall): {summary['total_tokens_all_time']}"
                )
                logger.info(
                    f"  Estimated Cost (Overall): ${summary['total_estimated_cost_all_time']:.4f}"
                )
            else:
                logger.info(
                    f"Token usage summary file is empty or invalid: {actual_summary_path}"
                )
        else:
            logger.info(
                f"Token usage summary file not found at: {actual_summary_path}"
            )
    except Exception as e:
        logger.error(f"Could not print token cost summary: {e}")


def run_batch_processing(args: Any, config: Dict[str, Any]):
    """Main function to run batch processing based on parsed arguments and configuration.
    Orchestrates helpers for file resolution, directory setup, processing, and reporting.
    """
    args.is_reprocessing_run = False # Initialize flag
    temp_reprocessing_input_path = None # To store path for cleanup

    # --- Reprocessing Setup ---
    if hasattr(args, 'reprocess_from') and args.reprocess_from:
        logger.info("--- Reprocessing Mode Activated ---")
        args.is_reprocessing_run = True
        try:
            # 1. Load previous output and original input
            logger.info(f"Loading previous output from: {args.reprocess_from}")
            previous_output_df = load_data(args.reprocess_from)
            if previous_output_df is None or previous_output_df.empty:
                logger.error(f"Failed to load or empty previous output file: {args.reprocess_from}. Aborting reprocessing.")
                return

            logger.info(f"Loading original input from: {args.reprocess_input_file}")
            original_input_df = load_data(args.reprocess_input_file)
            if original_input_df is None or original_input_df.empty:
                logger.error(f"Failed to load or empty original input file: {args.reprocess_input_file}. Aborting reprocessing.")
                return

            # 2. Identify failed items
            response_field = config.get('response_field_name', DEFAULT_RESPONSE_FIELD) # Match config_loader.py and file_processor.py
            id_column = args.original_input_id_column

            if response_field not in previous_output_df.columns:
                logger.error(f"Response field '{response_field}' not found in previous output file. Check config and file. Aborting.")
                return
            if id_column not in previous_output_df.columns:
                logger.error(f"ID column '{id_column}' not found in previous output file. Aborting.")
                return
            if id_column not in original_input_df.columns:
                logger.error(f"ID column '{id_column}' not found in original input file. Aborting.")
                return

            failed_items_output_df = previous_output_df[
                previous_output_df[response_field].astype(str).str.startswith("Error:", na=False)
            ]

            if failed_items_output_df.empty:
                logger.info("No items marked with 'Error:' in the response field found in the previous output. Nothing to reprocess.")
                return
            
            failed_ids = failed_items_output_df[id_column].unique().tolist()
            logger.info(f"Found {len(failed_ids)} unique items to reprocess based on ID column '{id_column}'.")

            items_to_reprocess_df = original_input_df[original_input_df[id_column].isin(failed_ids)]
            
            if items_to_reprocess_df.empty:
                logger.warning(f"No items from the original input file matched the failed IDs. IDs found: {failed_ids}. Double-check ID column and files.")
                return

            original_ext = Path(args.reprocess_input_file).suffix
            if not original_ext in ['.csv', '.jsonl', '.json']:
                 logger.warning(f"Original input file {args.reprocess_input_file} has an unsupported extension for reprocessing output. Defaulting to .csv for temp file.")
                 original_ext = '.csv'

            temp_dir = Path(tempfile.gettempdir()) / "batchgrader_reprocessing"
            temp_dir.mkdir(parents=True, exist_ok=True)
            # Store Path object for temp_reprocessing_input_path
            temp_reprocessing_input_path = temp_dir / f"reprocessing_input_{uuid.uuid4().hex}{original_ext}"
            
            save_data(items_to_reprocess_df, str(temp_reprocessing_input_path))
            logger.info(f"Prepared {len(items_to_reprocess_df)} items for reprocessing into temporary file: {temp_reprocessing_input_path}")

            args.input_file = str(temp_reprocessing_input_path)
            args.input_dir = None 
            logger.info(f"BatchGrader will now process only the failed items from: {args.input_file}")

        except Exception as e:
            logger.error(f"Error during reprocessing setup: {e}", exc_info=True)
            if temp_reprocessing_input_path and temp_reprocessing_input_path.exists():
                try:
                    temp_reprocessing_input_path.unlink()
                    logger.info(f"Cleaned up temporary reprocessing file due to setup error: {temp_reprocessing_input_path}")
                except OSError as e_clean:
                    logger.warning(f"Could not clean up temp file {temp_reprocessing_input_path} after setup error: {e_clean}")
            return
    
    # --- Main Processing Logic (wrapped for cleanup) ---
    try:
        log_dir, archive_dir = get_log_dirs(args)
        prune_logs_if_needed(log_dir, archive_dir, config=config)

        # Resolve files to process (this might have been updated by reprocessing logic)
        files_to_process = _resolve_files_to_process(args)
        if not files_to_process:
            # If it was a reprocessing run that resulted in no files, it's not an error, just nothing to do.
            # If it was a normal run, or reprocessing setup failed to produce a temp file, then it's potentially an issue, handled by _resolve_files_to_process logs.
            logger.info("No files to process at this stage.")
            return 

        output_directory = _setup_output_directory(args)
        if output_directory is None:
            logger.error("Failed to setup output directory. Aborting batch processing.")
            return

        processing_summary = _process_files(files_to_process, output_directory, config)
        _report_batch_results(processing_summary, files_to_process, args, log_dir, config)
    finally:
        # Cleanup temporary reprocessing file if it was created
        if args.is_reprocessing_run and temp_reprocessing_input_path and Path(temp_reprocessing_input_path).exists():
            try:
                Path(temp_reprocessing_input_path).unlink()
                logger.info(f"Successfully cleaned up temporary reprocessing file: {temp_reprocessing_input_path}")
            except OSError as e:
                logger.warning(f"Could not clean up temporary reprocessing file {temp_reprocessing_input_path}: {e}")


def _resolve_files_to_process(args) -> list:
    """
    Resolves and returns the list of files to process based on CLI args.
    """
    if getattr(args, 'input_file', None):
        return [args.input_file]
    elif getattr(args, 'input_dir', None):
        input_directory = Path(args.input_dir)
        if not input_directory.is_dir():
            logger.error(f"[ERROR] Input directory not found or not a directory: {input_directory}. Please verify the path and permissions.")
            return []
        files = [
            str(f) for f in input_directory.iterdir()
            if f.is_file() and f.name.endswith((".csv", ".xlsx", ".jsonl"))
        ]
        if not files:
            logger.warning(f"No suitable files (csv, xlsx, jsonl) found in directory: {input_directory}. Supported formats: .csv, .xlsx, .jsonl")
        return files
    else:
        logger.error("[ERROR] No input file or directory specified. Use --input-file or --input-dir to specify the input source.")
        return []


def _setup_output_directory(args) -> Optional[Path]:
    """
    Sets up and returns the output directory Path, or None on failure.
    """
    output_directory_str = getattr(args, 'output_dir', None) or str(Path.cwd() / "output" / "batch_results")
    output_directory = Path(output_directory_str)
    try:
        output_directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory set to: {output_directory}")
        return output_directory
    except Exception as e:
        logger.error(f"[ERROR] Failed to create output directory {output_directory}: {e}. Please verify permissions and path.")
        return None


def _process_files(files_to_process: list, output_directory: Path, config: dict) -> dict:
    """
    Processes each file and returns a result summary dict.
    """
    overall_success = True
    processed_files_count = 0
    failed_files_count = 0
    start_time = datetime.datetime.now()
    for filepath_str in files_to_process:
        filepath = Path(filepath_str)
        logger.info(f"\n--- Starting processing for: {filepath.name} (Size: {filepath.stat().st_size:,} bytes) ---")
        try:
            success = process_file(str(filepath), str(output_directory), config, args)
            if success:
                processed_files_count += 1
            else:
                failed_files_count += 1
                overall_success = False
                if not config.get("continue_on_failure", False):
                    logger.warning("Aborting batch early due to failure and continue_on_failure=False")
                    break
        except (IOError, OSError) as e:
            logger.error(
                f"File system error processing {filepath.name}: {str(e)}",
                exc_info=True,
                extra={
                    "file_info": {
                        "name": filepath.name,
                        "size": filepath.stat().st_size if filepath.exists() else None,
                        "last_modified": datetime.datetime.fromtimestamp(filepath.stat().st_mtime).isoformat() if filepath.exists() else None
                    },
                    "error_type": type(e).__name__,
                    "config": config
                })
            failed_files_count += 1
            overall_success = False
            if not config.get("continue_on_failure", False):
                logger.warning("Aborting batch early due to file system error and continue_on_failure=False")
                break
        except Exception as e:
            logger.error(
                f"Unhandled exception processing {filepath.name}: {str(e)}",
                exc_info=True,
                extra={
                    "file_info": {
                        "name": filepath.name,
                        "size": filepath.stat().st_size,
                        "last_modified": datetime.datetime.fromtimestamp(filepath.stat().st_mtime).isoformat()
                    },
                    "error_type": type(e).__name__,
                    "config": config
                })
            failed_files_count += 1
            overall_success = False
            if not config.get("continue_on_failure", False):
                logger.warning("Aborting batch early due to unhandled exception and continue_on_failure=False")
                break
        logger.info(f"--- Finished processing for: {filepath.name} ---")
    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds()
    return {
        "overall_success": overall_success,
        "processed_files_count": processed_files_count,
        "failed_files_count": failed_files_count,
        "duration": duration,
        "start_time": start_time,
        "end_time": end_time
    }


def _report_batch_results(result: dict, files_to_process: list, args: Any, log_dir: Path, config: dict):
    """
    Logs summary and prints stats if requested.
    """
    logger.info("\nBatch processing finished.")
    logger.info(f"Processing duration: {result['duration']:.2f} seconds ({result['duration']/60:.2f} minutes)")
    logger.info(f"Successfully processed files: {result['processed_files_count']}")
    logger.info(f"Failed files: {result['failed_files_count']}")
    if getattr(args, 'stats', False):
        print_token_cost_stats(config=config)
        print_token_cost_summary(log_dir=log_dir, config=config)
    if not result['overall_success'] and result['failed_files_count'] > 0:
        logger.warning(f"Some files failed to process ({result['failed_files_count']} out of {len(files_to_process)}). Please check the logs for details.")


def run_count_mode(args: Any, config: Dict[str, Any]):
    """Runs the token counting mode for input files."""
    logger.info("--- Running in COUNT mode ---")

    # Get log directories from args or constants
    log_dir, archive_dir = get_log_dirs(args)
    # No log pruning needed for count mode as it's typically non-invasive

    if args.input_file:
        files_to_process_str = [args.input_file]
    elif args.input_dir:
        input_directory = Path(args.input_dir)
        if not input_directory.is_dir():
            logger.error(
                f"[ERROR] Input directory not found or not a directory: {input_directory}"
            )
            return
        files_to_process_str = [
            str(f) for f in input_directory.iterdir()
            if f.is_file() and f.name.endswith(('.csv', '.xlsx', '.jsonl'))
        ]
        if not files_to_process_str:
            logger.warning(
                f"No suitable files (csv, xlsx, jsonl) found in directory: {input_directory}"
            )
            return
    else:
        logger.error(
            "[ERROR] No input file or directory specified for count mode.")
        return

    system_prompt_content = load_system_prompt(config)
    response_field = config.get('response_field_name', DEFAULT_RESPONSE_FIELD)

    # Get encoder using centralized utility
    model_name = config.get('openai_model_name', DEFAULT_MODEL)
    encoder = get_encoder(model_name)
    if encoder is None:
        logger.error("Failed to get encoder. Cannot perform token counting.")
        return

    total_files_counted = 0
    for filepath_str in files_to_process_str:
        filepath = Path(filepath_str)
        logger.info(f"Counting tokens for: {filepath.name}")
        try:
            df = load_data(str(filepath))
            if df.empty:
                logger.warning(
                    f"File {filepath.name} is empty. Skipping token count.")
                continue

            # Using check_token_limits just to get the stats dictionary.
            # The actual limit check (True/False) isn't the primary concern here, but the stats are.
            # We pass a very large token_limit to avoid triggering "limit exceeded" logs from check_token_limits.
            _is_valid, token_stats = check_token_limits(
                df,
                system_prompt_content,
                response_field,
                encoder,
                token_limit=sys.maxsize,
                filepath=filepath_str  # Pass the original filepath
            )

            if token_stats:  # If token_stats were successfully calculated
                logger.info(f"Token statistics for {filepath.name}:")
                logger.info(
                    f"  Total tokens: {token_stats.get('total', 'N/A')}")
                logger.info(
                    f"  Average tokens per row: {token_stats.get('average', 'N/A'):.2f}"
                )
                logger.info(
                    f"  Max tokens in a row: {token_stats.get('max', 'N/A')}")
                total_files_counted += 1
            else:
                logger.error(
                    f"Could not calculate token stats for {filepath.name}.")

        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
        except Exception as e:
            logger.error(f"Error counting tokens for {filepath.name}: {e}",
                         exc_info=True)

    logger.info(
        f"--- COUNT mode finished. Counted tokens for {total_files_counted} file(s). ---"
    )


def run_split_mode(args: Any, config: Dict[str, Any]):
    """Runs the file splitting mode for input files."""
    logger.info("--- Running in SPLIT mode ---")

    # Get log directories from args or constants
    log_dir, archive_dir = get_log_dirs(args)
    # No log pruning needed for split mode. Output is chunked files.

    if args.input_file:
        files_to_process_str = [args.input_file]
    elif args.input_dir:
        input_directory = Path(args.input_dir)
        if not input_directory.is_dir():
            logger.error(
                f"[ERROR] Input directory not found or not a directory: {input_directory}"
            )
            return
        files_to_process_str = [
            str(f) for f in input_directory.iterdir()
            if f.is_file() and f.name.endswith(('.csv', '.xlsx', '.jsonl'))
        ]
        if not files_to_process_str:
            logger.warning(
                f"No suitable files (csv, xlsx, jsonl) found in directory: {input_directory}"
            )
            return
    else:
        logger.error(
            "[ERROR] No input file or directory specified for split mode.")
        return

    system_prompt_content = load_system_prompt(config)
    response_field = config.get('response_field_name', DEFAULT_RESPONSE_FIELD)

    # Get encoder using centralized utility
    model_name = config.get('openai_model_name', DEFAULT_MODEL)
    encoder = get_encoder(model_name)
    if encoder is None:
        logger.error("Failed to get encoder. Cannot perform file splitting.")
        return

    # Splitting parameters from config or constants
    splitter_options = config.get('input_splitter_options', {})
    token_limit_per_chunk = splitter_options.get(
        'max_tokens_per_chunk',
        config.get('split_token_limit', DEFAULT_SPLIT_TOKEN_LIMIT))
    row_limit_per_chunk = splitter_options.get(
        'max_rows_per_chunk',
        config.get('split_row_limit'))  # Default to None if not specified
    force_chunk_count = splitter_options.get('force_chunk_count',
                                             None)  # Default to None

    # Output directory for chunks (defaults to input file's directory in a '_chunked' subfolder)
    # The `split_file_by_token_limit` function handles creating the _chunked dir.
    # We don't use args.output_dir here directly as split_file outputs relative to input.

    total_files_split = 0
    for filepath_str in files_to_process_str:
        filepath = Path(filepath_str)
        logger.info(f"Splitting file: {filepath.name} by token/row limits.")

        try:
            # The count_tokens_fn for the splitter needs to be defined.
            # It should take a row (pd.Series) and return token count.
            # We can reuse or adapt logic from create_token_counter or its internal workings.
            # For simplicity, let's use the one from _generate_chunk_job_objects in file_processor if possible or recreate.
            # Recreating here for clarity and independence:
            def _row_token_counter_for_splitter(row: pd.Series) -> int:
                row_content = " ".join(
                    str(value) for value in row.values if pd.notna(value))
                # This is a simplified version. If system_prompt is per row or complex, this needs adjustment.
                # For now, assume system_prompt is global and handled by batch job, not part of row data for splitting count.
                # Or, if we want the split to reflect system prompt usage PER CHUNK (which is more accurate for LLM calls):
                # This depends on how token_limit_per_chunk is meant to be interpreted.
                # Let's align with _generate_chunk_job_objects's _count_row_tokens which INCLUDES system prompt.
                full_content_for_tokenization = system_prompt_content + "\n" + row_content
                return len(encoder.encode(full_content_for_tokenization))

            chunk_paths, input_file_token_count = split_file_by_token_limit(
                input_path=str(filepath),
                token_limit=token_limit_per_chunk,
                count_tokens_fn=_row_token_counter_for_splitter,
                response_field=
                response_field,  # May not be strictly needed by splitter if count_tokens_fn is good
                row_limit=row_limit_per_chunk,
                force_chunk_count=force_chunk_count,
                output_dir=
                None,  # Let splitter use its default output dir (_chunked next to input)
                logger_override=logger  # Pass our logger
            )

            if chunk_paths:
                logger.success(
                    f"Successfully split {filepath.name} into {len(chunk_paths)} chunks:"
                )
                for chunk_path in chunk_paths:
                    logger.info(f"  - {chunk_path}")
                if input_file_token_count is not None:
                    logger.info(
                        f"Original file '{filepath.name}' estimated token count (by splitter): {input_file_token_count}"
                    )
                total_files_split += 1
            else:
                logger.warning(
                    f"File {filepath.name} was not split. It might be smaller than chunk limits or empty."
                )

        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
        except Exception as e:
            logger.error(f"Error splitting file {filepath.name}: {e}",
                         exc_info=True)

    logger.info(
        f"--- SPLIT mode finished. Split {total_files_split} file(s). ---")


if __name__ == "__main__":
    # Call the main function from the new cli.py module
    try:
        from src.cli import main as cli_main
        cli_main()
    except ImportError:
        # This fallback might be useful if running batch_runner.py directly in a way that messes with relative imports
        # However, the primary execution path should be via `python -m src.cli` or a setup.py entry point.
        logger.error(
            "Could not import cli.main. If running directly, ensure Python's import system can find src.cli."
        )
        logger.error("Try running: python -m src.cli [your_args]")
        # As a last resort for very direct script running for debugging, could try:
        # import cli
        # cli.main()
        # But this is not robust.
