"""
file_utils.py - BatchGrader File Utilities

Utility for pruning chunked input directories after batch jobs complete.
Deletes all files in the given _chunked directory except for .keep.
Safe for use after chunked job completion (original user inputs are never deleted).
"""
import os
from pathlib import Path


def prune_chunked_dir(chunked_dir: str):
    """
    Delete all files in chunked_dir except .keep. Directory is preserved.
    Args:
        chunked_dir (str): Path to the _chunked directory to prune.
    """
    p_chunked_dir = Path(chunked_dir)
    if not p_chunked_dir.is_dir():
        return
    for item in p_chunked_dir.iterdir():
        if item.is_file() and item.name != '.keep':
            item.unlink()
