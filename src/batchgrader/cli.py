import argparse
import logging
import sys
from pathlib import Path

from batchgrader.batch_runner import (
    run_batch_processing,
    run_count_mode,
    run_split_mode,
)
from batchgrader.config_loader import load_config
from batchgrader.constants import LOG_DIR as DEFAULT_LOG_DIR_CONSTANT
from batchgrader.constants import PROJECT_ROOT
from batchgrader.logger import setup_logging

logger = logging.getLogger(__name__)  # Define logger at module level


def main():
    """Main CLI entry point for BatchGrader."""

    # Determine initial log_dir for setup_logging, before full arg parsing
    # This is a bit of a workaround to get log_dir early.
    # A more robust solution might involve a pre-parsing step for --log-dir.
    pre_log_dir_val = None
    if "--log-dir" in sys.argv:
        from contextlib import suppress

        with suppress(ValueError):
            log_dir_index = sys.argv.index("--log-dir") + 1
            if log_dir_index < len(sys.argv):
                pre_log_dir_val = sys.argv[log_dir_index]

    # Use the extracted log_dir or default from constants for initial setup
    initial_log_dir = (Path(pre_log_dir_val)
                       if pre_log_dir_val else DEFAULT_LOG_DIR_CONSTANT)
    setup_logging(log_dir=initial_log_dir)

    parser = argparse.ArgumentParser(
        description=
        "BatchGrader CLI: Batch LLM evaluation, token counting, and input splitting."
    )

    # Input arguments
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument("--input-file",
                             type=str,
                             help="Path to a single input file to process.")
    input_group.add_argument(
        "--input-dir",
        type=str,
        help=
        "Path to a directory containing input files to process. If neither --input-file nor --input-dir is specified, the CLI will use the default input directory (input/) if it contains files.",
    )

    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=
        f"Directory for output files (default: {PROJECT_ROOT / 'output' / 'batch_results'}).",
    )

    # Configuration arguments
    parser.add_argument(
        "--config",
        dest="config_file",
        type=str,
        default=None,
        help=
        "Path to alternate config YAML file (default: config/config.yaml in project root).",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help=
        f"Directory for log files (default: {DEFAULT_LOG_DIR_CONSTANT} or as set by config).",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["batch", "count", "split"],
        default="batch",
        help=
        "Operation mode: 'batch' (process files), 'count' (count tokens), 'split' (split files by token limit). Default: batch",
    )

    # Other options
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show token/cost usage stats after processing.",
    )
    # parser.add_argument('--costs', action='store_true', help='DEPRECATED: Use --stats. Show token/cost usage stats and exit.') # Kept for compatibility, maybe remove later

    # Reprocessing arguments
    reprocessing_group = parser.add_argument_group(
        "Reprocessing Options",
        "Arguments for reprocessing failed items from a previous run.",
    )
    reprocessing_group.add_argument(
        "--reprocess-from",
        type=str,
        default=None,
        help=
        "Path to the output file from a previous run (e.g., .csv, .jsonl) containing failed items to reprocess.",
    )
    reprocessing_group.add_argument(
        "--reprocess-input-file",
        type=str,
        default=None,
        help=
        "Path to the *original* input file that was used to generate the output specified in --reprocess-from. Required if --reprocess-from is used.",
    )
    reprocessing_group.add_argument(
        "--original-input-id-column",
        type=str,
        default=None,
        help=
        "Name of the column in the --reprocess-input-file (and present in --reprocess-from output) that contains unique IDs for matching failed items. Required if --reprocess-from is used.",
    )

    args = parser.parse_args()

    # --- Validate reprocessing arguments ---
    _validate_reprocessing_args(parser, args, logger)

    # Default to input/ directory if neither input-file nor input-dir is provided
    # AND not in reprocessing mode (as reprocessing defines its input)
    if not args.input_file and not args.input_dir and not args.reprocess_from:
        from batchgrader.constants import PROJECT_ROOT as DYNAMIC_PROJECT_ROOT

        default_input_dir = DYNAMIC_PROJECT_ROOT / "input"
        if default_input_dir.exists() and any(default_input_dir.iterdir()):
            args.input_dir = str(default_input_dir)
            logger.info(
                f"No --input-file or --input-dir specified. Using default input directory: {args.input_dir}"
            )
        else:
            logger.error(
                "No --input-file or --input-dir specified, and the default input directory (input/) is missing or empty."
            )
            sys.exit(2)

    # Load configuration
    try:
        config = load_config(
            args.config_file)  # load_config handles default path if None
        logger.info(
            f"Configuration loaded successfully from: {args.config_file or 'default path'}"
        )
    except FileNotFoundError:
        logger.error(
            f"Configuration file not found: {args.config_file}. Using default configuration embedded in code."
        )
        config = {}  # Proceed with minimal/default config if file not found
    except ValueError as e:
        logger.error(
            f"Error loading configuration: {e}. Using default configuration embedded in code."
        )
        config = {}

    # --- Pass to batch_runner ---
    # run_batch_processing will handle log_dir override internally based on args.log_dir
    try:
        if args.mode == "batch":
            run_batch_processing(args, config)
        elif args.mode == "count":
            run_count_mode(args, config)
        elif args.mode == "split":
            run_split_mode(args, config)
        else:
            # This case should ideally not be reached if choices are enforced by argparse
            logger.error(f"Unsupported mode: {args.mode}")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user (Ctrl+C).")
        sys.exit(130)  # Standard exit code for ^C
    except SystemExit:
        # This allows sys.exit() to work as intended without being caught as a generic Exception
        # logger.debug(f"SystemExit called with code: {e.code}") # Optional: log if needed
        raise  # Re-raise to allow exit
    except Exception as e:
        logger.critical(
            f"A critical error occurred during {args.mode} mode processing: {e}",
            exc_info=True,
        )
        sys.exit(1)

    logger.info("CLI execution finished.")
    sys.exit(0)


def _validate_reprocessing_args(parser, args, logger):
    """
    Ensure required reprocessing arguments are provided if reprocessing mode is enabled.
    If --reprocess-from is specified, both --reprocess-input-file and --original-input-id-column are required.
    """
    if args.reprocess_from:
        missing = []
        if not args.reprocess_input_file:
            missing.append("--reprocess-input-file")
        if not args.original_input_id_column:
            missing.append("--original-input-id-column")
        if missing:
            logger.error(
                f"Missing required argument(s) for reprocessing: {', '.join(missing)}"
            )
            parser.error(
                "When --reprocess-from is specified, the following arguments are also required: "
                + ", ".join(missing))


if __name__ == "__main__":
    main()
