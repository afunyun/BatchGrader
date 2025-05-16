from pathlib import Path
import os


# Find the project root more reliably
# This finds the top-level directory that contains the src directory
def find_project_root() -> Path:
    """Find the project root directory from the current file."""
    # Start from the directory of this file and go up until we find the project root
    current_path = Path(__file__).resolve().parent

    # If this file is in src/, parent is the project root
    if current_path.name == 'src':
        return current_path.parent

    # Try to find the parent folders until we reach a folder that looks like the project root
    # i.e., contains both 'src' and 'tests' directories, or other key project files
    for parent in [current_path] + list(current_path.parents):
        # Check if this directory contains expected project root markers
        if (parent / 'src').exists() and (parent / 'tests').exists():
            return parent
        if (parent / 'pyproject.toml').exists() or (parent /
                                                    'setup.py').exists():
            return parent

    # If we couldn't find it based on structure, use a fallback based on file location
    # (removing 'src' from the path)
    return Path(__file__).resolve().parent.parent


# Project root directory
PROJECT_ROOT = find_project_root()

# Default paths for logs and archives relative to project root
# These are immutable default paths, code should not modify these at runtime
# Instead, pass directory paths as parameters to functions that need them
DEFAULT_LOG_DIR: Path = PROJECT_ROOT / 'output' / 'logs'
DEFAULT_ARCHIVE_DIR: Path = DEFAULT_LOG_DIR / 'archive'

# Paths for logs and archives (these can be updated by CLI args)
LOG_DIR: Path = DEFAULT_LOG_DIR
ARCHIVE_DIR: Path = DEFAULT_ARCHIVE_DIR

# Max rows per batch for non-chunked processing
MAX_BATCH_SIZE: int = 50000

# Default model name (fallback if not specified in config)
DEFAULT_MODEL: str = 'gpt-4o-mini-2024-07-18'

# Token Limits
DEFAULT_GLOBAL_TOKEN_LIMIT: int = 2_000_000
DEFAULT_SPLIT_TOKEN_LIMIT: int = 500_000

# Response field default (if not in config)
DEFAULT_RESPONSE_FIELD: str = "llm_response"

# Token Tracker Paths
DEFAULT_TOKEN_USAGE_LOG_PATH: Path = PROJECT_ROOT / 'output' / 'token_usage_log.json'
DEFAULT_EVENT_LOG_PATH: Path = PROJECT_ROOT / 'output' / 'token_usage_events.jsonl'
DEFAULT_PRICING_CSV_PATH: Path = PROJECT_ROOT / 'docs' / 'pricing.csv'

# Batch API Endpoint
BATCH_API_ENDPOINT: str = "/v1/chat/completions"

# Default prompts file path
DEFAULT_PROMPTS_FILE: Path = PROJECT_ROOT / 'config' / 'prompts.yaml'

# Default Batch Description
DEFAULT_BATCH_DESCRIPTION: str = "Batch API Job via BatchGrader"

# Default Poll Interval for checking batch job status (in seconds)
DEFAULT_POLL_INTERVAL: int = 60
