"""
Tracks submitted tokens by API key (censored) in output/token_usage_log.json.

"""
import os
import json
from datetime import datetime

LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output', 'token_usage_log.json')

def _get_api_key_prefix(api_key):
    if not api_key or len(api_key) < 10:
        return '****'
    return api_key[:10] + '**********'

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
    Add tokens to today's count for the given API key prefix.
    """
    if not date_str:
        date_str = datetime.now().strftime('%Y-%m-%d')
    prefix = _get_api_key_prefix(api_key)
    log = _load_log()
    # Find entry
    for entry in log:
        if entry['date'] == date_str and entry['api_key_prefix'] == prefix:
            entry['tokens_submitted'] += tokens_submitted
            _save_log(log)
            return
    # Not found, add new
    log.append({
        'date': date_str,
        'api_key_prefix': prefix,
        'tokens_submitted': tokens_submitted
    })
    _save_log(log)

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
