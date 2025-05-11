"""
file_utils.py - BatchGrader File Utilities

Utility for pruning chunked input directories after batch jobs complete.
Deletes all files in the given _chunked directory except for .keep.
Safe for use after chunked job completion (original user inputs are never deleted).
"""
import os

def prune_chunked_dir(chunked_dir):
    """
    Delete all files in chunked_dir except .keep. Directory is preserved.
    Args:
        chunked_dir (str): Path to the _chunked directory to prune.
    """
    if not os.path.isdir(chunked_dir):
        return
    for fname in os.listdir(chunked_dir):
        fpath = os.path.join(chunked_dir, fname)
        if os.path.isfile(fpath) and fname != '.keep':
            os.remove(fpath)
