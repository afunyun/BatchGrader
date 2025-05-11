"""
Utility for splitting input files into parts that do not exceed a specified token limit, preserving row integrity.
Specifically for batch_runner.py but could theoretically be used for other stuff.

Usage (from batch_runner.py):
    from input_splitter import split_file_by_token_limit
"""
import os
import pandas as pd

def split_file_by_token_limit(input_path, token_limit, count_tokens_fn, response_field, output_dir=None, file_prefix=None):
    """
    Args:
        input_path (str): Path to the input CSV/JSON/JSONL file.
        token_limit (int): Maximum allowed tokens per split file.
        count_tokens_fn (callable): Function to count tokens for a row.
        response_field (str): The field/column in the data containing the text to be evaluated.
        output_dir (str, optional): Directory to write split files. Defaults to input file's directory.
        file_prefix (str, optional): Prefix for output files. Defaults to input filename (no extension).
    Returns:
        List of output file paths.
    """
    if output_dir is None:
        output_dir = os.path.dirname(input_path)
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

    splits = []
    current_tokens = 0
    current_rows = []
    part_num = 1
    output_files = []

    for idx, row in df.iterrows():
        row_tokens = count_tokens_fn(row)
        
        if current_tokens + row_tokens > token_limit and current_rows:
            out_path = os.path.join(output_dir, f"{base_name}_part{part_num}{ext}")
            pd.DataFrame(current_rows).to_csv(out_path, index=False) if ext == '.csv' else pd.DataFrame(current_rows).to_json(out_path, orient='records', lines=(ext=='.jsonl'))
            output_files.append(out_path)
            part_num += 1
            current_rows = []
            current_tokens = 0
        current_rows.append(row)
        current_tokens += row_tokens
        
    if current_rows:
        out_path = os.path.join(output_dir, f"{base_name}_part{part_num}{ext}")
        pd.DataFrame(current_rows).to_csv(out_path, index=False) if ext == '.csv' else pd.DataFrame(current_rows).to_json(out_path, orient='records', lines=(ext=='.jsonl'))
        output_files.append(out_path)
    return output_files
