from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent

# Paths for logs and archives
LOG_DIR: Path = PROJECT_ROOT / 'output' / 'logs'
ARCHIVE_DIR: Path = LOG_DIR / 'archive'

# Max rows per batch for non-chunked processing
MAX_BATCH_SIZE: int = 50000

# Default model name (fallback if not specified in config)
DEFAULT_MODEL: str = 'gpt-4o-mini-2024-07-18' 