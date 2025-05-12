"""
Utility for splitting input files into parts that do not exceed a specified token or row limit, preserving row integrity.
Specifically for batch_runner.py but could theoretically be used for other stuff.

Usage (from batch_runner.py):
    from input_splitter import split_file_by_token_limit
    # Example usage:
    # output_files = split_file_by_token_limit(
    #     input_path, token_limit, count_tokens_fn, response_field,
    #     row_limit=row_limit, output_dir=..., file_prefix=...)

Configurable via config.yaml:
    split_token_limit: int (default: 500000)
    split_row_limit: int or null (default: unlimited)

Events:
    - input_split_config_loaded: {token_limit:int, row_limit:int}
    - file_split: {input_file:str, output_files:list}
    
Added force_chunk_count parameter to allow for fixed chunking.
Respected token_limit/row_limit as before.
"""
import os
import pandas as pd
from src.config_loader import load_config

def split_file_by_token_limit(input_path, token_limit=None, count_tokens_fn=None, response_field=None, row_limit=None, output_dir=None, file_prefix=None, force_chunk_count=None, logger=None):
    """
    Splits input file into chunks based on force_chunk_count (if set), otherwise token_limit/row_limit.
    - If force_chunk_count > 1: splits into N row-based chunks, then checks token count per chunk and warns/errors if any chunk exceeds token_limit.
    - Otherwise, splits by token_limit or row_limit as before.
    Logs mode and any warnings/errors.
    """
    import math
    import numpy as np
    import pandas as pd
    import os
    
    if output_dir is None:
        base_input_dir = os.path.dirname(input_path)
        output_dir = os.path.join(base_input_dir, '_chunked')
        os.makedirs(output_dir, exist_ok=True)
        keep_path = os.path.join(output_dir, '.keep')
        if not os.path.exists(keep_path):
            with open(keep_path, 'w', encoding='utf-8') as f:
                f.write('')
    base_name = file_prefix or os.path.splitext(os.path.basename(input_path))[0]
    ext = os.path.splitext(input_path)[1].lower()
    if ext == '.csv':
        df = pd.read_csv(input_path)
    elif ext == '.jsonl':
        df = pd.read_json(input_path, lines=True)
    elif ext == '.json':
        df = pd.read_json(input_path)
    else:
        raise ValueError(f"{ext} Is not supported for splitting.")

    output_files = []
    token_counts = []

    if force_chunk_count is not None and force_chunk_count > 1:
        if logger:
            logger.event(f"Chunking mode: force_chunk_count={force_chunk_count}")
        else:
            print(f"[EVENT] chunking_mode: force_chunk_count={force_chunk_count}")
        n_rows = len(df)
        chunk_sizes = [n_rows // force_chunk_count] * force_chunk_count
        for i in range(n_rows % force_chunk_count):
            chunk_sizes[i] += 1
        start = 0
        for part_num, size in enumerate(chunk_sizes, 1):
            chunk = df.iloc[start:start+size]
            start += size
            chunk_tokens = chunk.apply(count_tokens_fn, axis=1).sum() if count_tokens_fn else 0
            out_path = os.path.join(output_dir, f"{base_name}_part{part_num}{ext}")
            if ext == '.csv':
                chunk.to_csv(out_path, index=False)
            else:
                chunk.to_json(out_path, orient='records', lines=(ext=='.jsonl'))
            output_files.append(out_path)
            token_counts.append(chunk_tokens)
            if token_limit is not None and chunk_tokens > token_limit:
                msg = f"Chunk {part_num} exceeds token limit ({chunk_tokens} > {token_limit})!"
                if logger:
                    logger.warning(msg)
                else:
                    print(f"[WARN] {msg}")
        print(f"[EVENT] file_split: {{'input_file': '{input_path}', 'output_files': {output_files}, 'token_counts': {token_counts}}}")
        return output_files, token_counts

    current_tokens = 0
    current_rows = []
    part_num = 1
    for idx, row in df.iterrows():
        row_tokens = count_tokens_fn(row) if count_tokens_fn else 0
        if ((token_limit is not None and current_tokens + row_tokens > token_limit) or
            (row_limit is not None and len(current_rows) >= row_limit)) and current_rows:
            out_path = os.path.join(output_dir, f"{base_name}_part{part_num}{ext}")
            if ext == '.csv':
                pd.DataFrame(current_rows).to_csv(out_path, index=False)
            else:
                pd.DataFrame(current_rows).to_json(out_path, orient='records', lines=(ext=='.jsonl'))
            output_files.append(out_path)
            token_counts.append(current_tokens)
            part_num += 1
            current_rows = []
            current_tokens = 0
        current_rows.append(row)
        current_tokens += row_tokens
    if current_rows:
        out_path = os.path.join(output_dir, f"{base_name}_part{part_num}{ext}")
        if ext == '.csv':
            pd.DataFrame(current_rows).to_csv(out_path, index=False)
        else:
            pd.DataFrame(current_rows).to_json(out_path, orient='records', lines=(ext=='.jsonl'))
        output_files.append(out_path)
        token_counts.append(current_tokens)
    print(f"[EVENT] file_split: {{'input_file': '{input_path}', 'output_files': {output_files}, 'token_counts': {token_counts}}}")
    return output_files, token_counts

    """
    Args:
        input_path (str): Path to the input CSV/JSON/JSONL file.
        token_limit (int, optional): Maximum allowed tokens per split file. Loaded from config if not provided.
        count_tokens_fn (callable): Function to count tokens for a row.
        response_field (str): The field/column in the data containing the text to be evaluated.
        row_limit (int, optional): Maximum allowed rows per split file. Loaded from config if not provided.
        output_dir (str, optional): Directory to write split files. Defaults to input file's directory.
        file_prefix (str, optional): Prefix for output files. Defaults to input filename (no extension).
    Returns:
        Tuple: (list of output file paths, list of token counts per file)
    """
    config = load_config()
    if token_limit is None:
        token_limit = config.get('split_token_limit', 500_000)
    if row_limit is None:
        row_limit = config.get('split_row_limit', None)
    print(f"[EVENT] input_split_config_loaded: {{'token_limit': {token_limit}, 'row_limit': {row_limit}}}")

    if output_dir is None:
        base_input_dir = os.path.dirname(input_path)
        output_dir = os.path.join(base_input_dir, '_chunked')
        os.makedirs(output_dir, exist_ok=True)
        keep_path = os.path.join(output_dir, '.keep')
        if not os.path.exists(keep_path):
            with open(keep_path, 'w', encoding='utf-8') as f:
                f.write('')
    base_name = file_prefix or os.path.splitext(os.path.basename(input_path))[0]
    ext = os.path.splitext(input_path)[1].lower()
    if ext == '.csv':
        df = pd.read_csv(input_path)
    elif ext == '.jsonl':
        df = pd.read_json(input_path, lines=True)
    elif ext == '.json':
        df = pd.read_json(input_path)
    else:
        raise ValueError(f"{ext} Is not supported for splitting.")

    current_tokens = 0
    current_rows = []
    part_num = 1
    output_files = []
    token_counts = []

    split_tokens = 0
    for idx, row in df.iterrows():
        row_tokens = count_tokens_fn(row) if count_tokens_fn else 0
        if ((token_limit is not None and current_tokens + row_tokens > token_limit) or
            (row_limit is not None and len(current_rows) >= row_limit)) and current_rows:
            out_path = os.path.join(output_dir, f"{base_name}_part{part_num}{ext}")
            if ext == '.csv':
                pd.DataFrame(current_rows).to_csv(out_path, index=False)
            else:
                pd.DataFrame(current_rows).to_json(out_path, orient='records', lines=(ext=='.jsonl'))
            output_files.append(out_path)
            token_counts.append(current_tokens)
            part_num += 1
            current_rows = []
            current_tokens = 0
        current_rows.append(row)
        current_tokens += row_tokens
    if current_rows:
        out_path = os.path.join(output_dir, f"{base_name}_part{part_num}{ext}")
        if ext == '.csv':
            pd.DataFrame(current_rows).to_csv(out_path, index=False)
        else:
            pd.DataFrame(current_rows).to_json(out_path, orient='records', lines=(ext=='.jsonl'))
        output_files.append(out_path)
        token_counts.append(current_tokens)
    print(f"[EVENT] file_split: {{'input_file': '{input_path}', 'output_files': {output_files}, 'token_counts': {token_counts}}}")
    return output_files, token_counts
