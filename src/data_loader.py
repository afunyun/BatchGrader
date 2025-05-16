import pandas as pd
import json
import os
from pathlib import Path
from config_loader import load_config

def load_data(filepath):
    if filepath.endswith('.csv'):
        return pd.read_csv(filepath)
    elif filepath.endswith('.json'):
        return pd.read_json(filepath)
    elif filepath.endswith('.jsonl'):
        return pd.read_json(filepath, lines=True)
    else:
        raise ValueError("Unsupported file format")

def save_data(df, filepath):
    if filepath.endswith('.csv'):
        df.to_csv(filepath, index=False)
    elif filepath.endswith('.json'):
        df.to_json(filepath, orient='records')
    elif filepath.endswith('.jsonl'):
        df.to_json(filepath, orient='records', lines=True)
    else:
        raise ValueError("Unsupported file format")
