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
import os
import json
import csv
from src.config_loader import load_config
config = load_config()
from datetime import datetime
from typing import Optional, List, Dict, TextIO

LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output', 'token_usage_log.json')
EVENT_LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output', 'token_usage_events.jsonl')
PRICING_CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'docs', 'pricing.csv')

def _get_api_key_prefix(api_key):
    if not api_key or len(api_key) < 10:
        return '****'
    return api_key[:10] + '**********'

def _load_pricing() -> Dict[str, Dict[str, float]]:
    """
    Load model pricing from pricing.csv.
    Raises FileNotFoundError if the file is missing.
    Returns a dict of model -> {input, output} pricing.
    """
    if not os.path.exists(PRICING_CSV_PATH):
        raise FileNotFoundError(f"Pricing file not found: {PRICING_CSV_PATH}")
    pricing = {}
    with open(PRICING_CSV_PATH, 'r', encoding='utf-8') as f: # type: TextIO
        reader = csv.DictReader(f) # type: ignore
        for row in reader:
            model = row['Model']
            try:
                input_price = float(row['Input'])
                output_price = float(row['Output'])
            except (ValueError, KeyError) as e:
                logger.warning(f"Skipping pricing row due to error: {row}. Error: {e}")
                continue
            pricing[model] = {'input': input_price, 'output': output_price}
    return pricing

def _load_log():
    if not os.path.exists(LOG_PATH):
        return []
    try:
        with open(LOG_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return []

def _save_log(log):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=2)

def update_token_log(api_key, tokens_submitted, date_str=None):
    """
    Add tokens to today's count for the given API key prefix. (Legacy, for API limit tracking)
    """
    if not date_str:
        date_str = datetime.now().strftime('%Y-%m-%d')
    prefix = _get_api_key_prefix(api_key)
    log = _load_log()
    for entry in log:
        if entry['date'] == date_str and entry['api_key_prefix'] == prefix:
            entry['tokens_submitted'] += tokens_submitted
            _save_log(log)
            return
    log.append({
        'date': date_str,
        'api_key_prefix': prefix,
        'tokens_submitted': tokens_submitted
    })
    _save_log(log)

def log_token_usage_event(api_key: str, model: str, input_tokens: int, output_tokens: int, timestamp: Optional[str]=None, request_id: Optional[str]=None):
    """
    Log a single successful API request to the append-only event log (JSONL).
    Calculates cost using pricing table (per 1M tokens). Only call after a successful response.
    """
    pricing = _load_pricing()
    if model not in pricing:
        input_price = output_price = 0.0
    else:
        input_price = pricing[model]['input']
        output_price = pricing[model]['output']
        
    cost = (input_tokens * input_price + output_tokens * output_price) / 1_000_000.0
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
    os.makedirs(os.path.dirname(EVENT_LOG_PATH), exist_ok=True)
    with open(EVENT_LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(json.dumps(event) + '\n')

def load_token_usage_events() -> List[Dict]:
    """
    Load all token usage events from the append-only log.
    """
    if not os.path.exists(EVENT_LOG_PATH):
        return []
    with open(EVENT_LOG_PATH, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def get_token_usage_summary(start_date: Optional[str]=None, end_date: Optional[str]=None, group_by: str='day') -> Dict:
    """
    Aggregate token usage and cost over a date range. group_by: 'day'|'model'|'all'.
    Returns: { 'total_tokens': int, 'total_cost': float, 'breakdown': Dict }
    """
    events = load_token_usage_events()
    summary = {'total_tokens': 0, 'total_cost': 0.0, 'breakdown': {}}
    def in_range(ts):
        dt = datetime.fromisoformat(ts[:19])
        if start_date and dt.date() < datetime.fromisoformat(start_date).date():
            return False
        if end_date and dt.date() > datetime.fromisoformat(end_date).date():
            return False
        return True
    for e in events:
        if not in_range(e['timestamp']):
            continue
        summary['total_tokens'] += e['total_tokens']
        summary['total_cost'] += e['cost']
        key = None
        if group_by == 'day':
            key = e['timestamp'][:10]
        elif group_by == 'model':
            key = e['model']
        elif group_by == 'all':
            key = 'all'
        if key:
            if key not in summary['breakdown']:
                summary['breakdown'][key] = {'tokens': 0, 'cost': 0.0, 'count': 0}
            summary['breakdown'][key]['tokens'] += e['total_tokens']
            summary['breakdown'][key]['cost'] += e['cost']
            summary['breakdown'][key]['count'] += 1
    summary['total_cost'] = round(summary['total_cost'], 6)
    for v in summary['breakdown'].values():
        v['cost'] = round(v['cost'], 6)
    return summary

def get_total_cost(start_date: Optional[str]=None, end_date: Optional[str]=None) -> float:
    """
    Return total cost for the given date range (inclusive).
    """
    return get_token_usage_summary(start_date, end_date, group_by='all')['total_cost']

def get_token_usage_for_day(api_key, date_str=None):
    """
    Return tokens submitted for this key/date.
    """
    if not date_str:
        date_str = datetime.now().strftime('%Y-%m-%d')
    prefix = _get_api_key_prefix(api_key)
    log = _load_log()
    for entry in log:
        if entry['date'] == date_str and entry['api_key_prefix'] == prefix:
            return entry['tokens_submitted']
    return 0
