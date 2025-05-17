"""
BatchGraderLogger: Configuration for application-wide logging.
- Console output uses rich.logging.RichHandler for colorized, pretty logs.
- File logs are written with standard formatting for post-mortem/debug.
- Call setup_logging() at application entry point.
"""

import logging
from datetime import datetime
from pathlib import Path

from rich.logging import RichHandler

SUCCESS_LEVEL_NUM = 25
logging.addLevelName(SUCCESS_LEVEL_NUM, "SUCCESS")


def success(self, message, *args, **kws):
    """Logs a 'SUCCESS' level message on this logger.

    The 'SUCCESS' level (25) is numerically between INFO (20) and WARNING (30).
    This method is added to the `logging.Logger` class.

    Args:
        message: The message to log.
        *args: Arguments to be merged into msg.
        **kws: Keyword arguments for the logging machinery.
    """
    # Yes, logger takes its '*args' as 'args'.
    if self.isEnabledFor(SUCCESS_LEVEL_NUM):
        self._log(SUCCESS_LEVEL_NUM, message, args, **kws)


logging.Logger.success = success


def setup_logging(
        log_dir: Path = Path("output/logs"), log_level: int = logging.INFO):
    """Configures the application's root logger for console and file output.

    - Clears any existing handlers on the root logger.
    - Sets the root logger's level to `log_level`.
    - Adds a `RichHandler` for colorful console output.
    - Adds a `FileHandler` for writing logs to a timestamped file in `log_dir`.
      The log file will be named `batchgrader_run_{timestamp}.log`.
    - Both handlers are set to `log_level`.

    Args:
        log_dir: The directory where log files will be stored.
                 Defaults to 'output/logs' relative to where script is run if not specified.
                 Can be a string path or a Path object.
        log_level: The minimum logging level for the handlers (e.g., logging.INFO, logging.DEBUG).
                   Defaults to logging.INFO.
    """
    # Ensure log_dir is a Path object by calling Path() on it.
    # If log_dir is already a Path instance, Path(log_dir) is idempotent.
    # If log_dir is a string, it will be converted.
    resolved_log_dir = Path(log_dir)

    resolved_log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = resolved_log_dir / f"batchgrader_run_{timestamp}.log"

    # Get the root logger, or a specific application base logger like logging.getLogger('batchgrader')
    # For simplicity, configuring the root logger here.
    # If you want to avoid affecting other libraries' logging, use a named base logger.
    app_logger = logging.getLogger()  # Get root logger

    # Clear existing handlers from the root logger to avoid duplicates if this is called multiple times
    # or if other parts of the code/libraries also configure the root logger.
    if app_logger.hasHandlers():
        app_logger.handlers.clear()

    app_logger.setLevel(log_level)

    console_handler = RichHandler(
        show_time=True,
        show_level=True,
        show_path=False,  # Usually False for cleaner output
        rich_tracebacks=True,
        log_time_format="[%X]",
    )  # Example time format
    console_handler.setLevel(log_level)

    file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
    file_formatter = logging.Formatter(
        "[%(asctime)s] %(name)s:%(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(log_level)

    app_logger.addHandler(console_handler)
    app_logger.addHandler(file_handler)

    # Initial log message to confirm setup
    app_logger.info(f"Logging setup complete. Log file: {log_file}")
