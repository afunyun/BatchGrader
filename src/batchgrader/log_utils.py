"""
log_utils.py - BatchGrader Log Pruning Utility

Provides utilities for log pruning/archiving to prevent unbounded growth of the logs directory.

- Moves oldest logs to archive when threshold is exceeded
- Deletes oldest archive logs if archive is full
- Logs all actions to prune.log for auditability
- Designed for use at BatchGrader startup, before logger instantiation

Configurable constants:
- MAX_LOGS: Max logs to keep in output/logs/
- MAX_ARCHIVE: Max logs to keep in output/logs/archive/
"""

import shutil
from datetime import datetime
from pathlib import Path


def prune_logs_if_needed(log_dir,
                         archive_dir,
                         max_logs=None,
                         max_archive=None,
                         config=None):
    """
    Prune logs in log_dir if over max_logs.
    Moves oldest logs to archive_dir. If archive_dir exceeds max_archive, deletes oldest in archive_dir.
    Logs all actions to prune.log in archive_dir.

    Args:
        log_dir: Directory where log files are stored (str or Path)
        archive_dir: Directory where archived log files will be moved (str or Path)
        max_logs: Maximum number of log files to keep in log_dir before archiving (default from config)
        max_archive: Maximum number of log files to keep in archive_dir (default from config)
        config: Configuration dictionary that may contain 'max_logs' and 'max_archive' keys
    """
    # Use provided values or extract from config or use defaults
    config = config or {}
    max_logs_val = max_logs or config.get("max_logs", 100)
    max_archive_val = max_archive or config.get("max_archive", 500)
    if (not isinstance(max_logs_val, int) or max_logs_val <= 0
            or not isinstance(max_archive_val, int) or max_archive_val <= 0):
        raise ValueError("max_logs and max_archive must be positive integers")

    p_log_dir = Path(log_dir)
    p_archive_dir = Path(archive_dir)

    p_log_dir.mkdir(parents=True, exist_ok=True)
    p_archive_dir.mkdir(parents=True, exist_ok=True)
    prune_log_file = p_archive_dir / "prune.log"

    def log_action(action, files):
        with open(prune_log_file, "a", encoding="utf-8") as f:
            ts = datetime.now().isoformat()
            f.write(f"[{ts}] {action.upper()}: {files}\n")

    # Process logs in log_dir
    current_logs = [
        item for item in p_log_dir.iterdir()
        if item.is_file() and item.name != ".keep"
        and item.name != "archive"  # 'archive' check in case it's a file
    ]
    current_logs.sort(key=lambda f: f.stat().st_mtime)

    if len(current_logs) > max_logs_val:
        to_move = current_logs[:len(current_logs) - max_logs_val]
        moved_files_log = []
        for file_path in to_move:
            try:
                destination = p_archive_dir / file_path.name
                shutil.move(str(file_path), str(destination))
                moved_files_log.append(file_path.name)
            except Exception as e:
                # Log error if move fails, but continue
                log_action(
                    "move_error",
                    f"{file_path.name} to {p_archive_dir / file_path.name}: {e}",
                )
        if moved_files_log:
            log_action("move", moved_files_log)

    # Process logs in archive_dir
    archived_logs = [
        item for item in p_archive_dir.iterdir()
        if item.is_file() and item.name != "prune.log"
    ]
    archived_logs.sort(key=lambda f: f.stat().st_mtime)

    if len(archived_logs) > max_archive_val:
        to_delete = archived_logs[:len(archived_logs) - max_archive_val]
        deleted_files_log = []
        for file_path in to_delete:
            try:
                file_path.unlink()  # Replaces os.remove
                deleted_files_log.append(file_path.name)
            except Exception as e:
                # Log error if delete fails, but continue
                log_action("delete_error", f"{file_path.name}: {e}")
        if deleted_files_log:
            log_action("delete", deleted_files_log)
