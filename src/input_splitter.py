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
"""
import os
import pandas as pd
from config_loader import load_config

def split_file_by_token_limit(input_path, token_limit=None, count_tokens_fn=None, response_field=None, row_limit=None, output_dir=None, file_prefix=None):
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
