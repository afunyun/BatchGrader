import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


from batchgrader.constants import (
    DEFAULT_GLOBAL_TOKEN_LIMIT,
    DEFAULT_RESPONSE_FIELD,
)

# load_data and save_data were here, removed as unused by current functions
from batchgrader.file_processor import (
    process_file_wrapper,
)
from batchgrader.llm_client import LLMClient

# Retain prune_logs_if_needed for now, may be used by run_batch_processing
from batchgrader.prompt_utils import load_system_prompt
from batchgrader.log_utils import prune_logs_if_needed

logger = logging.getLogger(__name__)


def get_log_dirs(args: Any) -> Tuple[Path, Path]:
    """
    Get log and archive directories, either from args or using defaults from constants.

    Args:
        args: Command line arguments which may include log_dir

    Returns:
        Tuple of (log_dir, archive_dir) as Path objects
    """
    from batchgrader.constants import DEFAULT_ARCHIVE_DIR, DEFAULT_LOG_DIR

    if hasattr(args, "log_dir") and args.log_dir:
        log_dir = Path(args.log_dir).resolve()
        archive_dir = log_dir / "archive"
        logger.info(f"Log directory from CLI: {log_dir}")
        logger.info(f"Archive directory: {archive_dir}")
    else:
        log_dir = DEFAULT_LOG_DIR
        archive_dir = DEFAULT_ARCHIVE_DIR

    return log_dir, archive_dir


def process_file(
    filepath_str: str,
    output_dir_str: str,
    config: Dict[str, Any],
    args: Optional[Any] = None,
) -> bool:
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

    try:
        result = _process_file_with_config(config, filepath, output_dir, args)
    except FileNotFoundError as _e:
        logger.error(
            f"[ERROR] Input file not found: {filepath}. Please verify the file exists and you have read permissions."
        )
        result = False
    except ValueError as _e:
        logger.error(
            f"[ERROR] Configuration or input error for {filepath}: {_e}. Current config: {config}"
        )
        result = False
    except RuntimeError as _e:
        logger.critical(
            f"[CRITICAL ERROR] Unexpected error processing {filepath}: {_e}"
        )
        result = False

    return result


def _process_file_with_config(
    config: Dict[str, Any],
    filepath: Path,
    output_dir: Path,
    args: Optional[Any] = None,
) -> bool:
    """
    Processes a single file using the new file_processor.process_file_wrapper.
    This function is intended to be used only by process_file, and is not intended for external use.

    Args:
        config: Configuration dictionary containing processing parameters
        filepath: Path to the input file
        output_dir: Output directory path
        args: Command line arguments which may include log_dir (default None)

    Returns:
        bool: True if processing was successful, False otherwise
    """
    system_prompt_content = load_system_prompt(config)
    response_field = config.get("response_field_name", DEFAULT_RESPONSE_FIELD)

    # Get encoder via LLMClient, as tests mock this pathway.
    # LLMClient's __init__ handles getting the model_name and initializing the encoder.
    temp_llm_client = LLMClient(config=config)
    encoder = temp_llm_client.encoder

    token_limit = config.get("global_token_limit", DEFAULT_GLOBAL_TOKEN_LIMIT)
    is_reprocessing_run = (
        args.reprocess if args and hasattr(args, "reprocess") else False
    )

    # process_file_wrapper is imported from batchgrader.file_processor
    return process_file_wrapper(
        filepath=str(filepath),
        output_dir=str(output_dir),
        config=config,
        system_prompt_content=system_prompt_content,
        response_field=response_field,
        encoder=encoder,
        token_limit=token_limit,
        is_reprocessing_run=is_reprocessing_run,
    )


def run_batch_processing(args: Any, config: Dict[str, Any]):
    log_dir, archive_dir = get_log_dirs(args)  # Get log_dir and archive_dir
    prune_logs_if_needed(log_dir, archive_dir)

    files_to_process = []
    if args.input_file:
        files_to_process.append(Path(args.input_file))
    elif args.input_dir:
        input_path = Path(args.input_dir)
        # Basic glob for common data file types, can be expanded
        for ext in ("*.csv", "*.jsonl", "*.json"):
            files_to_process.extend(list(input_path.glob(ext)))
        if not files_to_process:
            logger.warning(
                f"No files found in input directory: {args.input_dir}")
            return
    else:
        logger.error("No input file or directory specified.")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    continue_on_failure = config.get("continue_on_failure", False)

    for filepath in files_to_process:
        logger.info(f"Starting processing for: {filepath.name}")
        success = process_file(str(filepath), str(output_dir), config, args)
        if not success:
            logger.error(f"Failed to process file: {filepath.name}")
            if not continue_on_failure:
                logger.error("Halting batch processing due to failure.")
                break
        else:
            logger.info(f"Successfully processed: {filepath.name}")


def run_count_mode():
    pass


def run_split_mode():

    pass


# Closing bracket removed to resolve syntax error
