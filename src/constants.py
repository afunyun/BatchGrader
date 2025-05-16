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
