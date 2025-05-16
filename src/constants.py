from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

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
