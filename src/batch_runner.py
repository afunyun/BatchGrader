from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from config_loader import load_config
from constants import (
    DEFAULT_GLOBAL_TOKEN_LIMIT,
    DEFAULT_MODEL,
    DEFAULT_RESPONSE_FIELD,
    DEFAULT_SPLIT_TOKEN_LIMIT,
    MAX_BATCH_SIZE,
)
from data_loader import load_data, save_data
from file_processor import check_token_limits, prepare_output_path, process_file_wrapper
from input_splitter import split_file_by_token_limit
from llm_client import LLMClient
from logger import logger
from log_utils import prune_logs_if_needed
from prompt_utils import load_system_prompt
from token_tracker import get_token_usage_for_day, get_token_usage_summary, log_token_usage_event, update_token_log


def get_log_dirs(args) -> Tuple[Path, Path]:
    """
    Get log and archive directories, either from args or using defaults from constants.
    
    Args:
        args: Command line arguments which may include log_dir
        
    Returns:
        Tuple of (log_dir, archive_dir) as Path objects
    """
    from constants import DEFAULT_LOG_DIR, DEFAULT_ARCHIVE_DIR

    if hasattr(args, 'log_dir') and args.log_dir:
        log_dir = Path(args.log_dir).resolve()
        archive_dir = log_dir / 'archive'
        logger.info(f"Log directory from CLI: {log_dir}")
        logger.info(f"Archive directory: {archive_dir}")
    else:
        log_dir = DEFAULT_LOG_DIR
        archive_dir = DEFAULT_ARCHIVE_DIR

    return log_dir, archive_dir


def process_file(filepath_str: str, output_dir_str: str,
                 config: Dict[str, Any]) -> bool:
    """
    Processes a single file using the new file_processor.process_file_wrapper.
    This function now acts as a simpler interface to the core processing logic.
    """
    filepath = Path(filepath_str)
    output_dir = Path(output_dir_str)
    logger.info(f"Initiating processing for: {filepath.name} -> {output_dir}")

    try:
        system_prompt_content = load_system_prompt(config)
        response_field = config.get('response_field_name',
                                    DEFAULT_RESPONSE_FIELD)

        # Get encoder (tiktoken)
        # This might be better suited inside file_processor or llm_client if always the same
        encoder = None
        try:
            # Pass config to LLMClient instead of having it load config itself
            llm_client_for_encoder = LLMClient(config=config)
            encoder = llm_client_for_encoder.encoder
            if encoder is None:
                raise ValueError("LLMClient did not provide an encoder.")
        except Exception as e:
            logger.error(
                f"Failed to get encoder: {e}. Token counting and splitting might be affected."
            )
            # Decide if we should proceed without an encoder or raise an error / return False
            # For now, we'll let process_file_wrapper handle it, it might have fallbacks or raise.

        # Use global token limit from constants, overridden by config if present
        token_limit = config.get('global_token_limit',
                                 DEFAULT_GLOBAL_TOKEN_LIMIT)

        # Delegate to the centralized file processing wrapper
        success = process_file_wrapper(
            filepath=str(
                filepath
            ),  # process_file_wrapper might still expect string paths
            output_dir=str(output_dir),
            config=config,
            system_prompt_content=system_prompt_content,
            response_field=response_field,
            encoder=encoder,
            token_limit=token_limit)

        if success:
            logger.success(f"Successfully processed {filepath.name}.")
        else:
            logger.error(
                f"Failed to process {filepath.name}. See logs for details.")
        return success

    except FileNotFoundError:
        logger.error(f"[ERROR] Input file not found: {filepath}")
        return False
    except ValueError as ve:
        logger.error(
            f"[ERROR] Configuration or input error for {filepath.name}: {ve}")
        return False
    except Exception as e:
        logger.error(
            f"[CRITICAL ERROR] Unexpected error processing {filepath.name}: {e}",
            exc_info=True)
        # Log to a specific error file for this run might be good if not already handled by process_file_wrapper
        return False


def print_token_cost_stats(config=None):
    """Prints token usage statistics for the current day."""
    try:
        # Pass config to LLMClient instead of having it load config itself
        llm_client = LLMClient(config=config)
        api_key = llm_client.api_key
        daily_usage = get_token_usage_for_day(api_key)
        if daily_usage:
            logger.info("Token Usage Stats (Today):")
            logger.info(f"  Total Tokens: {daily_usage['total_tokens']}")
            logger.info(
                f"  Estimated Cost: ${daily_usage['estimated_cost']:.4f}")
            # Add more details if available in daily_usage
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
            from constants import DEFAULT_LOG_DIR
            log_dir = DEFAULT_LOG_DIR

        # Default path if not provided
        default_summary_path = log_dir / f"token_usage_summary_{api_key[:8]}.json"
        actual_summary_path = Path(
            summary_file_path) if summary_file_path else default_summary_path

        if actual_summary_path.exists():
            summary = get_token_usage_summary(
                str(actual_summary_path
                    ))  # get_token_usage_summary might expect str
            if summary:
                logger.info("Overall Token Usage Summary (from file):")
                logger.info(
                    f"  Total Tokens (Overall): {summary['total_tokens_all_time']}"
                )
                logger.info(
                    f"  Estimated Cost (Overall): ${summary['total_estimated_cost_all_time']:.4f}"
                )
                # Add more details as available
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
    """
    Main function to run batch processing based on parsed arguments and configuration.
    (CLI parsing will be moved to cli.py, this function will be called by cli.py)
    """
    # Get log directories from args or constants
    log_dir, archive_dir = get_log_dirs(args)

    # Now that log_dir and archive_dir are determined (not modifying globals), prune logs
    prune_logs_if_needed(log_dir, archive_dir, config=config)

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
            "[ERROR] No input file or directory specified. Use --input-file or --input-dir."
        )
        return

    output_directory_str = args.output_dir if args.output_dir else str(
        Path.cwd() / "output" / "batch_results")  # Default output
    output_directory = Path(output_directory_str)
    output_directory.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory set to: {output_directory}")

    overall_success = True
    processed_files_count = 0
    failed_files_count = 0

    for filepath_str in files_to_process_str:
        filepath = Path(filepath_str)
        logger.info(f"\n--- Starting processing for: {filepath.name} ---")
        try:
            success = process_file(str(filepath), str(output_directory),
                                   config)
            if success:
                processed_files_count += 1
            else:
                failed_files_count += 1
                overall_success = False  # If any file fails, overall is not a complete success
        except Exception as e:
            logger.error(
                f"Unhandled exception processing {filepath.name}: {str(e)}",
                exc_info=True)
            failed_files_count += 1
            overall_success = False
        logger.info(f"--- Finished processing for: {filepath.name} ---")

    logger.info("\nBatch processing finished.")
    logger.info(f"Successfully processed files: {processed_files_count}")
    logger.info(f"Failed files: {failed_files_count}")

    if args.stats:
        print_token_cost_stats(config=config)
        print_token_cost_summary(log_dir=log_dir, config=config)

    if not overall_success and failed_files_count > 0:
        logger.warning("Some files failed to process. Please check the logs.")


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

    # Get encoder for token counting
    encoder = None
    try:
        llm_client_for_encoder = LLMClient(config=config)
        encoder = llm_client_for_encoder.encoder
        if encoder is None:
            raise ValueError(
                "LLMClient did not provide an encoder for token counting.")
    except Exception as e:
        logger.error(
            f"Failed to get encoder: {e}. Cannot perform token counting.")
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
                token_limit=float('inf'))

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

    # Get encoder (tiktoken)
    encoder = None
    try:
        llm_client_for_encoder = LLMClient()
        encoder = llm_client_for_encoder.encoder
        if encoder is None:
            raise ValueError(
                "LLMClient did not provide an encoder for splitting.")
    except Exception as e:
        logger.error(
            f"Failed to get encoder: {e}. Cannot perform file splitting.")
        return

    # Splitting parameters from config or constants
    splitter_options = config.get('input_splitter_options', {})
    token_limit_per_chunk = splitter_options.get(
        'max_tokens_per_chunk',
        config.get('split_token_limit', DEFAULT_SPLIT_TOKEN_LIMIT))
    row_limit_per_chunk = splitter_options.get(
        'max_rows_per_chunk',
        config.get('split_row_limit',
                   None))  # Default to None if not specified
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
        from cli import main as cli_main
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
