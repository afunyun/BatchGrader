"""
Token Tracker: Tracks and aggregates OpenAI API token usage for both API limit enforcement and historical/cost tracking.

- Daily aggregate for API limit compliance (legacy, output/token_usage_log.json)
- Per-request append-only event log (output/token_usage_events.jsonl)
- Aggregation and cost computation utilities

Event Schema (token_usage_event):
    {
    "timestamp": ISO8601,
    "api_key_prefix": str,
    "model": str,
    "input_tokens": int,
    "output_tokens": int,
    "total_tokens": int,
    "cost": float,
    "request_id": str (optional)
    }

Cost is calculated using model pricing from docs/pricing.csv (per 1M tokens, input/output).
"""
import csv
import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, TextIO, Any

import pandas as pd
from src.input_splitter import FileNotFoundError

logger = logging.getLogger(__name__)


def _get_api_key_prefix(api_key: Optional[str]):
    """Generates a masked prefix for an API key for logging purposes.

    If the key is shorter than 10 characters, or None/empty, returns '****'.
    Otherwise, returns the first 10 characters followed by '**********'.

    Args:
        api_key: The API key string.

    Returns:
        A masked string representation of the API key's prefix.
    """
    if not api_key or len(api_key) < 10:
        return '****'
    # Make sure to return exactly 10 characters followed by 10 asterisks
    return api_key[:10] + '**********'


def _load_pricing(pricing_csv_path: Path) -> Dict[str, Dict[str, float]]:
    """
    Load model pricing from pricing.csv.
    Raises FileNotFoundError if the file is missing.
    Returns a dict of model -> {input, output} pricing.
    """
    if not pricing_csv_path.exists():
        raise FileNotFoundError(
            f"Pricing file not found: {pricing_csv_path}")
    pricing = {}
    with open(pricing_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)  # type: ignore
        for row in reader:
            model = row['Model']
            try:
                input_price = float(row['Input'])
                output_price = float(row['Output'])
            except (ValueError, KeyError) as e:
                logger.warning(
                    f"Skipping pricing row due to error: {row}. Error: {e}")
                continue
            pricing[model] = {'input': input_price, 'output': output_price}
    return pricing


def _load_log(log_path: Path) -> List[Dict[str, Any]]:
    """Loads a JSON log file into a list.

    If the file doesn't exist or an error occurs during loading (e.g., invalid JSON),
    an empty list is returned.

    Args:
        log_path: Path to the JSON log file.

    Returns:
        A list containing the log entries, or an empty list if loading fails.
    """
    if not log_path.exists():
        return []
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to load JSON log: {e}")
        return []
    except Exception as e:
        logger.error(f"An error occurred while loading log: {e}")
        return []


def _save_log(log: List[Dict[str, Any]], log_path: Path):
    """Saves a list of log entries to a JSON file.

    Ensures the parent directory for the log file exists.
    The JSON is saved with an indent of 2 for readability.

    Args:
        log: A list of log entries to save.
        log_path: Path to the JSON log file where entries will be saved.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=2)


def update_token_log(api_key: str,
                     tokens_submitted: int,
                     log_path: Path,
                     date_str: Optional[str] = None):
    """
    Add tokens to today's count for the given API key prefix. (Legacy, for API limit tracking)
    """
    if not date_str:
        date_str = datetime.now().strftime('%Y-%m-%d')
    prefix = _get_api_key_prefix(api_key)
    log = _load_log(log_path)
    for entry in log:
        if entry['date'] == date_str and entry['api_key_prefix'] == prefix:
            entry['tokens_submitted'] += tokens_submitted
            _save_log(log, log_path)
            return
    log.append({
        'date': date_str,
        'api_key_prefix': prefix,
        'tokens_submitted': tokens_submitted
    })
    _save_log(log, log_path)


def log_token_usage_event(api_key: str,
                          model: str,
                          input_tokens: int,
                          output_tokens: int,
                          event_log_path: Path,
                          pricing_csv_path: Path,
                          timestamp: Optional[str] = None,
                          request_id: Optional[str] = None):
    """
    Log a single successful API request to the append-only event log (JSONL).
    Calculates cost using pricing table (per 1M tokens). Only call after a successful response.
    """
    pricing = _load_pricing(pricing_csv_path)
    if model not in pricing:
        input_price = output_price = 0.0
    else:
        input_price = pricing[model]['input']
        output_price = pricing[model]['output']

    cost = (input_tokens * input_price +
            output_tokens * output_price) / 1_000_000.0
    event = {
        'timestamp': timestamp or datetime.now().isoformat(),
        'api_key_prefix': _get_api_key_prefix(api_key),
        'model': model,
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'total_tokens': input_tokens + output_tokens,
        'cost': round(cost, 6)
    }
    if request_id:
        event['request_id'] = request_id
    event_log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(event_log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(event) + '\n')


def load_token_usage_events(event_log_path: Path) -> List[Dict]:
    """
    Load all token usage events from the append-only log.
    """
    if not event_log_path.exists():
        return []
    with open(event_log_path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


def get_token_usage_summary(event_log_path: Path,
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None,
                            group_by: str = 'day') -> Dict:
    """
    Aggregate token usage and cost over a date range. group_by: 'day'|'model'|'all'.
    Returns: { 'total_tokens': int, 'total_cost': float, 'breakdown': Dict }
    """
    events = load_token_usage_events(event_log_path)
    summary = {'total_tokens': 0, 'total_cost': 0.0, 'breakdown': {}}

    def in_range(ts):
        dt = datetime.fromisoformat(ts[:19])
        if start_date and dt.date() < datetime.fromisoformat(
                start_date).date():
            return False
        if end_date and dt.date() > datetime.fromisoformat(end_date).date():
            return False
        return True

    for e in events:
        if not in_range(e['timestamp']):
            continue
        summary['total_tokens'] += e['total_tokens']
        summary['total_cost'] += e['cost']
        
        key_to_set = None
        if group_by == 'day':
            key_to_set = e['timestamp'][:10]
        elif group_by == 'model':
            key_to_set = e['model']
        elif group_by == 'all':
            key_to_set = 'all_time_summary' # Consistent key for 'all'
        # else: key_to_set remains None if group_by is unrecognized

        if key_to_set:
            if key_to_set not in summary['breakdown']:
                summary['breakdown'][key_to_set] = {'tokens': 0, 'cost': 0.0}
            summary['breakdown'][key_to_set]['tokens'] += e['total_tokens']
            summary['breakdown'][key_to_set]['cost'] += e['cost']
            summary['breakdown'][key_to_set].setdefault('count', 0)
            summary['breakdown'][key_to_set]['count'] += 1
    summary['total_cost'] = round(summary['total_cost'], 6)
    for v in summary['breakdown'].values():
        v['cost'] = round(v['cost'], 6)
    return summary


def get_total_cost(event_log_path: Path,
                   start_date: Optional[str] = None,
                   end_date: Optional[str] = None) -> float:
    """
    Return total cost for the given date range (inclusive).
    """
    return get_token_usage_summary(event_log_path,
                                   start_date,
                                   end_date,
                                   group_by='all')['total_cost']


def get_token_usage_for_day(api_key: str,
                            log_path: Path,
                            date_str: Optional[str] = None) -> int:
    """
    Return tokens submitted for this key/date.
    """
    if not date_str:
        date_str = datetime.now().strftime('%Y-%m-%d')

    # Generate the prefix in the same way it was stored
    prefix = _get_api_key_prefix(api_key)
    log = _load_log(log_path)

    return next((
        entry['tokens_submitted']
        for entry in log
        if entry['date'] == date_str and entry['api_key_prefix'] == prefix
    ), 0)
