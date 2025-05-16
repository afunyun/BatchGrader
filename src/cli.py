import argparse
import sys
from pathlib import Path
from config_loader import load_config
# Import specific mode functions from batch_runner
from batch_runner import run_batch_processing, run_count_mode, run_split_mode
from logger import logger # Use the centralized logger
from constants import PROJECT_ROOT, LOG_DIR as DEFAULT_LOG_DIR # For default log dir path

# To ensure logger is configured before first use if batch_runner's top-level logger setup is removed/changed.
# However, batch_runner.py itself might still initialize it. This is a bit tricky.
# For now, assume logger is available.

def main():
    """Main CLI entry point for BatchGrader."""
    parser = argparse.ArgumentParser(description="BatchGrader CLI: Batch LLM evaluation, token counting, and input splitting.")
    
    # Input arguments
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input-file", type=str, help="Path to a single input file to process.")
    input_group.add_argument("--input-dir", type=str, help="Path to a directory containing input files to process.")

    # Output arguments
    parser.add_argument("--output-dir", type=str, default=None, help=f"Directory for output files (default: {PROJECT_ROOT / 'output' / 'batch_results'}).")

    # Configuration arguments
    parser.add_argument("--config", dest='config_file', type=str, default=None, help="Path to alternate config YAML file (default: config/config.yaml in project root).")
    parser.add_argument("--log-dir", type=str, default=None, help=f"Directory for log files (default: {DEFAULT_LOG_DIR} or as set by config).")
    
    # Operational mode arguments (though batch_runner currently assumes 'batch' mode mostly)
    parser.add_argument("--mode", type=str, choices=['batch', 'count', 'split'], default='batch', help="Operation mode: 'batch' (process files), 'count' (count tokens), 'split' (split files by token limit). Default: batch")
    # TODO: Implement actual count/split modes based on this arg in batch_runner or file_processor

    # Other options
    parser.add_argument("--stats", action="store_true", help="Show token/cost usage stats after processing.")
    # parser.add_argument('--costs', action='store_true', help='DEPRECATED: Use --stats. Show token/cost usage stats and exit.') # Kept for compatibility, maybe remove later

    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.config_file) # load_config handles default path if None
        logger.info(f"Configuration loaded successfully from: {args.config_file if args.config_file else 'default path'}")
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {args.config_file}. Using default configuration embedded in code.")
        config = {} # Proceed with minimal/default config if file not found
    except ValueError as e:
        logger.error(f"Error loading configuration: {e}. Using default configuration embedded in code.")
        config = {}

    # --- Pass to batch_runner --- 
    # run_batch_processing will handle log_dir override internally based on args.log_dir
    try:
        if args.mode == 'batch':
            run_batch_processing(args, config)
        elif args.mode == 'count':
            run_count_mode(args, config)
        elif args.mode == 'split':
            run_split_mode(args, config)
        else:
            # This case should ideally not be reached if choices are enforced by argparse
            logger.error(f"Unsupported mode: {args.mode}")
            sys.exit(1)
            
    except Exception as e:
        logger.critical(f"A critical error occurred during {args.mode} mode processing: {e}", exc_info=True)
        sys.exit(1)
    
    logger.info("CLI execution finished.")
    sys.exit(0)

if __name__ == '__main__':
    main() 