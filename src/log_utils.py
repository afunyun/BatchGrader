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
import os
import shutil
from datetime import datetime
from pathlib import Path

from config_loader import load_config

config = load_config()

MAX_LOGS = config['max_logs']
MAX_ARCHIVE = config['max_archive']


def prune_logs_if_needed(log_dir,
                         archive_dir,
                         max_logs=MAX_LOGS,
                         max_archive=MAX_ARCHIVE):
    """
    Prune logs in log_dir if over max_logs.
    Moves oldest logs to archive_dir. If archive_dir exceeds max_archive, deletes oldest in archive_dir.
    Logs all actions to prune.log in archive_dir.
    """
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(archive_dir, exist_ok=True)
    prune_log = os.path.join(archive_dir, 'prune.log')

    def log_action(action, files):
        with open(prune_log, 'a', encoding='utf-8') as f:
            ts = datetime.now().isoformat()
            f.write(f"[{ts}] {action.upper()}: {files}\n")

    logs = [
        f for f in os.listdir(log_dir)
        if os.path.isfile(os.path.join(log_dir, f)) and f != '.keep'
        and f != 'archive'
    ]
    logs.sort(key=lambda f: os.path.getmtime(os.path.join(log_dir, f)))

    if len(logs) > max_logs:
        to_move = logs[:len(logs) - max_logs]
        for filename in to_move:
            src = os.path.join(log_dir, filename)
            dst = os.path.join(archive_dir, filename)
            shutil.move(src, dst)
        log_action('move', to_move)

    archived = [
        f for f in os.listdir(archive_dir)
        if os.path.isfile(os.path.join(archive_dir, f)) and f != 'prune.log'
    ]
    archived.sort(key=lambda f: os.path.getmtime(os.path.join(archive_dir, f)))
    if len(archived) > max_archive:
        to_delete = archived[:len(archived) - max_archive]
        for filename in to_delete:
            os.remove(os.path.join(archive_dir, filename))
        log_action('delete', to_delete)
