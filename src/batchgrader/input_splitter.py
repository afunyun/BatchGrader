"""
Utility for splitting input files into parts that do not exceed a specified token or row limit, preserving row integrity.
Specifically for batch_runner.py but could theoretically be used for other stuff.

Usage (from batch_runner.py):
    # Import the function
    from input_splitter import split_file_by_token_limit

    # Use function with token counter and limits
    chunks, token_counts = split_file_by_token_limit(
        input_path="data.csv",
        token_limit=5000,
        count_tokens_fn=my_token_counter
    )

Configurable via config.yaml:
    split_token_limit: int (default: 500000)
    split_row_limit: int or null (default: unlimited)

Events:
    - input_split_config_loaded: {token_limit:int, row_limit:int}
    - file_split: {input_file:str, output_files:list}

Added force_chunk_count parameter to allow for fixed chunking.
Respected token_limit/row_limit as before.
"""

import builtins
import logging
import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import pandas as pd

# Define module-level logger BEFORE it's used in default args or function bodies
logger = logging.getLogger(__name__)


class InputSplitterError(Exception):
    """Base exception for input splitter errors"""

    pass


class MissingArgumentError(InputSplitterError):
    """Exception raised when a required argument is missing"""

    pass


class FileNotFoundError(builtins.FileNotFoundError, InputSplitterError):
    """Exception raised when input file is not found"""

    pass


class UnsupportedFileTypeError(InputSplitterError):
    """Exception raised when file type is not supported"""

    pass


class OutputDirectoryError(InputSplitterError):
    """Exception raised when output directory cannot be determined or created"""

    pass


def split_file_by_token_limit(
    input_path: Optional[str],
    token_limit: Optional[int] = None,
    count_tokens_fn: Optional[Callable[[pd.Series], int]] = None,
    response_field: Optional[str] = None,
    row_limit: Optional[int] = None,
    output_dir: Optional[str] = None,
    file_prefix: Optional[str] = None,
    force_chunk_count: Optional[int] = None,
    logger_override: Optional[logging.Logger] = None,
    df: Optional[pd.DataFrame] = None,
    _original_ext: Optional[str] = None,
) -> Tuple[List[str], List[int]]:
    """
    Splits input file or DataFrame into chunks based on force_chunk_count (if set), otherwise token_limit/row_limit.
    - If force_chunk_count > 1: splits into N row-based chunks, then checks token count per chunk and warns/errors if any chunk exceeds token_limit.
    - Otherwise, splits by token_limit or row_limit as before.
    If df is provided, it is used as the input data; otherwise, input_path is loaded.

    Args:
        input_path: Path to the input file to split
        token_limit: Maximum number of tokens per chunk
        count_tokens_fn: Function to count tokens in a row
        response_field: Name of the response field in the data
        row_limit: Maximum number of rows per chunk
        output_dir: Directory to save split files
        file_prefix: Prefix for output file names
        force_chunk_count: If > 1, forcibly split into this many chunks
        logger_override: Logger instance to use for this function call
        df: Optional DataFrame to use instead of loading from file
        _original_ext: Original file extension (internal use for recursive calls)

    Returns:
        Tuple of (output_files, token_counts) where output_files is a list of file paths and token_counts is a list of token counts.

    Raises:
        MissingArgumentError: When a required argument like file_prefix is missing in recursive calls
        FileNotFoundError: When input file is not found
        UnsupportedFileTypeError: When file type is not supported
        OutputDirectoryError: When output directory cannot be determined
        ValueError: When neither token_limit nor row_limit is provided, or when token_limit is set but count_tokens_fn is missing
    """
    current_logger = (logger_override
                      or logger)  # Use module-level logger if override is None

    current_logger.debug(
        f"Splitting file: {input_path or 'DataFrame input'}, Token Limit: {token_limit}, Row Limit: {row_limit}, Force Chunks: {force_chunk_count}"
    )

    # Validate that at least one of token_limit, row_limit, or force_chunk_count is set
    if token_limit is None and row_limit is None and force_chunk_count is None:
        raise ValueError(
            "At least one of token_limit, row_limit, or force_chunk_count must be provided"
        )

    # Validate that if token_limit is set, count_tokens_fn is provided
    if token_limit is not None and count_tokens_fn is None:
        raise ValueError(
            "count_tokens_fn must be provided when using token_limit")

    # Validate that if force_chunk_count is set, it's a positive integer
    if force_chunk_count is not None and (
            not isinstance(force_chunk_count, int) or force_chunk_count < 1):
        raise ValueError(
            f"force_chunk_count must be a positive integer, got {force_chunk_count}"
        )

    # Validate that if row_limit is set, it's a positive integer
    if row_limit is not None and (not isinstance(row_limit, int)
                                  or row_limit < 1):
        raise ValueError(
            f"row_limit must be a positive integer, got {row_limit}")

    current_ext = None
    current_base_name = None

    if _original_ext:
        current_ext = _original_ext

        if file_prefix:
            current_base_name = file_prefix
        else:
            raise MissingArgumentError(
                "file_prefix must be provided in recursive calls when _original_ext is set."
            )
    elif input_path:
        p_input_path = Path(input_path)
        current_ext = p_input_path.suffix.lower()
        current_base_name = file_prefix or p_input_path.stem
    elif df is None:
        raise MissingArgumentError(
            "Cannot determine file type or name: input_path is None, _original_ext is not set, and df is None."
        )
    else:
        raise MissingArgumentError(
            "Cannot determine output file type: df provided without input_path or _original_ext."
        )

    if df is not None:
        loaded_df = df
    else:
        p_input_path = Path(input_path)
        if not p_input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        if current_ext == ".csv":
            loaded_df = pd.read_csv(input_path)
        elif current_ext == ".jsonl":
            loaded_df = pd.read_json(input_path, lines=True)
        elif current_ext == ".json":
            loaded_df = pd.read_json(input_path)
        else:
            raise UnsupportedFileTypeError(
                f"Unsupported file type: {current_ext} for file {input_path}")

    if loaded_df.empty:
        current_logger.warning(
            f"Input data is empty for {'DataFrame processing' if input_path is None else input_path}."
        )
        return [], []

    if output_dir is None:
        if input_path:
            p_input_path = Path(input_path)
            base_input_dir = p_input_path.parent
            p_output_dir = base_input_dir / "_chunked"
        else:
            raise OutputDirectoryError(
                "output_dir is None and cannot be defaulted as input_path was not provided."
            )
    else:
        p_output_dir = Path(output_dir)

    p_output_dir.mkdir(parents=True, exist_ok=True)
    keep_file_path = p_output_dir / ".keep"
    if not keep_file_path.exists():
        with open(keep_file_path, "w", encoding="utf-8") as f:
            f.write("")

    current_base_name = os.path.basename(current_base_name)
    len(loaded_df)
    output_files = []
    token_counts = []

    if force_chunk_count is not None and force_chunk_count > 1:
        current_logger.info(
            f"Chunking mode: force_chunk_count={force_chunk_count}")
        n_rows = len(loaded_df)
        chunk_sizes = [n_rows // force_chunk_count] * force_chunk_count
        for i in range(n_rows % force_chunk_count):
            chunk_sizes[i] += 1
        start = 0
        temp_output_files = []
        temp_token_counts = []
        for part_num, size in enumerate(chunk_sizes, 1):
            chunk = loaded_df.iloc[start:start + size]
            start += size
            chunk_tokens = (chunk.apply(count_tokens_fn, axis=1).sum()
                            if count_tokens_fn else 0)
            if token_limit is not None and chunk_tokens > token_limit:
                current_logger.warning(
                    f"Chunk {part_num} ({len(chunk)} rows) for '{current_base_name}' exceeds token limit ({chunk_tokens} > {token_limit}), recursively splitting."
                )

                recursive_file_prefix = f"{current_base_name}_part{part_num}_split"

                chunk_out_files, chunk_token_counts = ((
                    [], []) if chunk.empty else split_file_by_token_limit(
                        input_path=None,
                        token_limit=token_limit,
                        count_tokens_fn=count_tokens_fn,
                        response_field=response_field,
                        row_limit=None,
                        output_dir=output_dir,
                        file_prefix=recursive_file_prefix,
                        force_chunk_count=None,
                        logger_override=current_logger,
                        df=None if chunk.empty else chunk,
                        _original_ext=current_ext,
                    ))
                temp_output_files.extend(chunk_out_files)
                temp_token_counts.extend(chunk_token_counts)
            else:
                out_path_p = (
                    p_output_dir /
                    f"{current_base_name}_part{part_num}{current_ext}")
                if current_ext == ".csv":
                    chunk.to_csv(str(out_path_p), index=False)
                else:
                    chunk.to_json(
                        str(out_path_p),
                        orient="records",
                        lines=(current_ext == ".jsonl"),
                    )
                temp_output_files.append(str(out_path_p))
                temp_token_counts.append(chunk_tokens)
        output_files = temp_output_files
        token_counts = temp_token_counts
        log_input_ref = input_path or "DataFrame"
        current_logger.info(
            f"File split: {log_input_ref} into {len(output_files)} chunks with token counts {token_counts}"
        )
        return output_files, token_counts

    current_tokens = 0
    current_rows = []
    part_num = 1

    for row_tuple in loaded_df.itertuples(index=False):
        # Convert namedtuple to Series for compatibility with existing code
        row = pd.Series(row_tuple, index=loaded_df.columns)
        row_tokens = count_tokens_fn(row) if count_tokens_fn else 0

        if current_rows and (
            (token_limit is not None
             and current_tokens + row_tokens > token_limit) or
            (row_limit is not None and len(current_rows) >= row_limit)):
            out_path_p = (p_output_dir /
                          f"{current_base_name}_part{part_num}{current_ext}")
            if current_ext == ".csv":
                pd.DataFrame(current_rows).to_csv(str(out_path_p), index=False)
            else:
                pd.DataFrame(current_rows).to_json(
                    str(out_path_p),
                    orient="records",
                    lines=(current_ext == ".jsonl"))
            output_files.append(str(out_path_p))
            token_counts.append(current_tokens)
            part_num += 1
            current_rows = []
            current_tokens = 0
        current_rows.append(row)
        current_tokens += row_tokens

    if current_rows:
        out_path_p = p_output_dir / \
            f"{current_base_name}_part{part_num}{current_ext}"
        if current_ext == ".csv":
            pd.DataFrame(current_rows).to_csv(str(out_path_p), index=False)
        else:
            pd.DataFrame(current_rows).to_json(str(out_path_p),
                                               orient="records",
                                               lines=(current_ext == ".jsonl"))
        output_files.append(str(out_path_p))
        token_counts.append(current_tokens)

    log_input_ref = input_path or "DataFrame"
    split_mode = "token_limit" if token_limit is not None else "row_limit"
    limit_val = token_limit if token_limit is not None else row_limit
    current_logger.info(
        f"File split: {log_input_ref} using {split_mode}={limit_val} into {len(output_files)} chunks"
    )
    return output_files, token_counts
