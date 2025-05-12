"""
Utility for splitting input files into parts that do not exceed a specified token or row limit, preserving row integrity.
Specifically for batch_runner.py but could theoretically be used for other stuff.

Usage (from batch_runner.py):
    from input_splitter import split_file_by_token_limit
    
    
    
    

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

def split_file_by_token_limit(input_path, token_limit=None, count_tokens_fn=None, response_field=None, row_limit=None, output_dir=None, file_prefix=None, force_chunk_count=None, logger=None, df=None, _original_ext=None):
    """
    Splits input file or DataFrame into chunks based on force_chunk_count (if set), otherwise token_limit/row_limit.
    - If force_chunk_count > 1: splits into N row-based chunks, then checks token count per chunk and warns/errors if any chunk exceeds token_limit.
    - Otherwise, splits by token_limit or row_limit as before.
    If df is provided, it is used as the input data; otherwise, input_path is loaded.
    """
    import math
    import numpy as np
    import pandas as pd
    import os

    print(f"[DEBUG_SPLITTER_ENTRY] Initial force_chunk_count: {force_chunk_count}, token_limit: {token_limit}, row_limit: {row_limit}") # Debug print at entry
    if logger:
        logger.debug(f"Splitting file: {input_path if input_path else 'DataFrame input'}, Token Limit: {token_limit}, Row Limit: {row_limit}, Force Chunks: {force_chunk_count}")
    else:
        print(f"[DEBUG] Splitting file: {input_path if input_path else 'DataFrame input'}, Token Limit: {token_limit}, Row Limit: {row_limit}, Force Chunks: {force_chunk_count}")

    current_ext = None
    current_base_name = None

    if _original_ext:
        current_ext = _original_ext
        
        if not file_prefix:
            if logger: logger.error("file_prefix must be provided in recursive calls when _original_ext is set.")
            print("[ERROR] file_prefix must be provided in recursive calls.")
            return [], []
        current_base_name = file_prefix 
    elif input_path:
        current_ext = os.path.splitext(input_path)[1].lower()
        current_base_name = file_prefix or os.path.splitext(os.path.basename(input_path))[0]
    else:
        
        
        if df is None:
            if logger: logger.error("Cannot determine file type or name: input_path is None, _original_ext is not set, and df is None.")
            print("[ERROR] Cannot determine file type or name: insufficient information.")
            return [], []
        else:
            
            
            if logger: logger.error("Cannot determine output file type: df provided without input_path or _original_ext.")
            print("[ERROR] Cannot determine output file type for provided DataFrame without input_path or explicit original extension.")
            return [], []

    if df is not None:
        loaded_df = df
    else: 
        if not input_path or not os.path.exists(input_path):
            if logger: logger.error(f"Input file not found: {input_path}")
            else: print(f"[ERROR] Input file not found: {input_path}")
            return [], []

        if current_ext == '.csv':
            loaded_df = pd.read_csv(input_path)
        elif current_ext == '.jsonl':
            loaded_df = pd.read_json(input_path, lines=True)
        elif current_ext == '.json':
            loaded_df = pd.read_json(input_path)
        else:
            if logger: logger.error(f"Unsupported file type: {current_ext} for file {input_path}")
            else: print(f"[ERROR] Unsupported file type: {current_ext} for file {input_path}")
            return [], []

    if loaded_df.empty:
        if logger: logger.warning(f"Input data is empty for {'DataFrame processing' if input_path is None else input_path}.")
        else: print(f"[WARN] Input data is empty for {'DataFrame processing' if input_path is None else input_path}.")
        return [], []

    if output_dir is None:
        if input_path: 
            base_input_dir = os.path.dirname(input_path)
            output_dir = os.path.join(base_input_dir, '_chunked')
        else: 
            if logger: logger.error("output_dir is None and cannot be defaulted as input_path was not provided.")
            print("[ERROR] output_dir must be specified when input_path is not provided.")
            return [], []
            
    os.makedirs(output_dir, exist_ok=True)
    keep_path = os.path.join(output_dir, '.keep')
    if not os.path.exists(keep_path):
        with open(keep_path, 'w', encoding='utf-8') as f:
            f.write('')
            
    current_base_name = os.path.basename(current_base_name) 
    total_rows = len(loaded_df)
    output_files = []
    token_counts = []
    
    if force_chunk_count is not None and force_chunk_count > 1:
        print(f"[DEBUG_SPLITTER] Entered force_chunk_count block. force_chunk_count = {force_chunk_count}, input_path = {input_path}") # Debug print
        if logger:
            logger.event(f"Chunking mode: force_chunk_count={force_chunk_count}")
        else:
            print(f"[EVENT] chunking_mode: force_chunk_count={force_chunk_count}")
        n_rows = len(loaded_df)
        chunk_sizes = [n_rows // force_chunk_count] * force_chunk_count
        for i in range(n_rows % force_chunk_count):
            chunk_sizes[i] += 1
        start = 0
        temp_output_files = []
        temp_token_counts = []
        for part_num, size in enumerate(chunk_sizes, 1):
            chunk = loaded_df.iloc[start:start+size]
            start += size
            chunk_tokens = chunk.apply(count_tokens_fn, axis=1).sum() if count_tokens_fn else 0
            if token_limit is not None and chunk_tokens > token_limit:
                if logger:
                    logger.warning(f"Chunk {part_num} ({len(chunk)} rows) for '{current_base_name}' exceeds token limit ({chunk_tokens} > {token_limit}), recursively splitting.")
                else:
                    print(f"[WARN] Chunk {part_num} ({len(chunk)} rows) for '{current_base_name}' exceeds token limit ({chunk_tokens} > {token_limit}), recursively splitting.")
                
                recursive_file_prefix = f"{current_base_name}_part{part_num}_split"
                
                chunk_out_files, chunk_token_counts = split_file_by_token_limit(
                    input_path=None, 
                    token_limit=token_limit,
                    count_tokens_fn=count_tokens_fn,
                    response_field=response_field,
                    row_limit=None, 
                    output_dir=output_dir,
                    file_prefix=recursive_file_prefix,
                    force_chunk_count=None, 
                    logger=logger,
                    df=chunk if not chunk.empty else None,
                    _original_ext=current_ext 
                ) if not chunk.empty else ([], [])
                temp_output_files.extend(chunk_out_files)
                temp_token_counts.extend(chunk_token_counts)
            else:
                out_path = os.path.join(output_dir, f"{current_base_name}_part{part_num}{current_ext}")
                if current_ext == '.csv':
                    chunk.to_csv(out_path, index=False)
                else: 
                    chunk.to_json(out_path, orient='records', lines=(current_ext=='.jsonl'))
                temp_output_files.append(out_path)
                temp_token_counts.append(chunk_tokens)
        output_files = temp_output_files
        token_counts = temp_token_counts
        log_input_ref = input_path if input_path else "DataFrame"
        event_msg = f"file_split: {{'input_file': '{log_input_ref}', 'split_mode': 'force_chunk_count', 'num_chunks': {force_chunk_count}, 'output_files': {output_files}, 'token_counts': {token_counts}}}"
        if logger: logger.event(event_msg)
        else: print(f"[EVENT] {event_msg}")
        return output_files, token_counts
    
    current_tokens = 0
    current_rows = []
    part_num = 1
    
    for idx, row in loaded_df.iterrows():
        row_tokens = count_tokens_fn(row) if count_tokens_fn else 0
        if ((token_limit is not None and current_tokens + row_tokens > token_limit) or
            (row_limit is not None and len(current_rows) >= row_limit)) and current_rows:
            out_path = os.path.join(output_dir, f"{current_base_name}_part{part_num}{current_ext}")
            if current_ext == '.csv':
                pd.DataFrame(current_rows).to_csv(out_path, index=False)
            else:
                pd.DataFrame(current_rows).to_json(out_path, orient='records', lines=(current_ext=='.jsonl'))
            output_files.append(out_path)
            token_counts.append(current_tokens)
            part_num += 1
            current_rows = []
            current_tokens = 0
        current_rows.append(row)
        current_tokens += row_tokens
    if current_rows:
        out_path = os.path.join(output_dir, f"{current_base_name}_part{part_num}{current_ext}")
        if current_ext == '.csv':
            pd.DataFrame(current_rows).to_csv(out_path, index=False)
        else:
            pd.DataFrame(current_rows).to_json(out_path, orient='records', lines=(current_ext=='.jsonl'))
        output_files.append(out_path)
        token_counts.append(current_tokens)
    log_input_ref = input_path if input_path else "DataFrame"
    split_mode = 'token_limit' if token_limit else 'row_limit'
    limit_val = token_limit if token_limit else row_limit
    event_msg = f"file_split: {{'input_file': '{log_input_ref}', 'split_mode': '{split_mode}', 'limit_value': {limit_val}, 'output_files': {output_files}, 'token_counts': {token_counts}}}"
    if logger: logger.event(event_msg)
    else: print(f"[EVENT] {event_msg}")

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
