"""
Unit tests for the token_tracker module.
"""

import json
import os
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock, call

from src.token_tracker import (_get_api_key_prefix, _load_pricing, _load_log,
                               _save_log, update_token_log,
                               log_token_usage_event, load_token_usage_events,
                               get_token_usage_summary, get_total_cost,
                               get_token_usage_for_day)


@pytest.fixture
def mock_pricing_csv():
    """Mock pricing CSV data for tests."""
    return "Model,Input,Output\n" \
           "gpt-4o-mini-2024-07-18,0.15,0.60\n" \
           "gpt-4o-2024-08-06,1.00,3.00\n"


@pytest.fixture
def mock_token_log():
    """Mock token usage log for tests."""
    return [{
        "date": "2025-05-10",
        "api_key_prefix": "sk-1234567**********",
        "tokens_submitted": 5000
    }, {
        "date": "2025-05-11",
        "api_key_prefix": "sk-1234567**********",
        "tokens_submitted": 10000
    }, {
        "date": "2025-05-11",
        "api_key_prefix": "sk-abcdefg**********",
        "tokens_submitted": 7500
    }]


@pytest.fixture
def mock_token_events():
    """Mock token usage events for tests."""
    return [
        json.dumps({
            "timestamp": "2025-05-10T10:00:00",
            "api_key_prefix": "sk-1234567890**********",
            "model": "gpt-4o-mini-2024-07-18",
            "input_tokens": 1000,
            "output_tokens": 500,
            "total_tokens": 1500,
            "cost": 0.00045
        }),
        json.dumps({
            "timestamp": "2025-05-11T11:00:00",
            "api_key_prefix": "sk-1234567890**********",
            "model": "gpt-4o-2024-08-06",
            "input_tokens": 2000,
            "output_tokens": 1000,
            "total_tokens": 3000,
            "cost": 0.005
        }),
        json.dumps({
            "timestamp": "2025-05-11T12:00:00",
            "api_key_prefix": "sk-abcdefghij**********",
            "model": "gpt-4o-mini-2024-07-18",
            "input_tokens": 3000,
            "output_tokens": 1500,
            "total_tokens": 4500,
            "cost": 0.00135
        })
    ]


def test_get_api_key_prefix():
    """Test extracting API key prefix."""
    # Full key case
    assert _get_api_key_prefix(
        "sk-1234567890abcdefghijklmn") == "sk-1234567**********"

    # Short key case
    assert _get_api_key_prefix("short") == "****"

    # None case
    assert _get_api_key_prefix(None) == "****"

    # Empty string case
    assert _get_api_key_prefix("") == "****"


def test_load_pricing(mock_pricing_csv):
    """Test loading pricing data from CSV."""
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data=mock_pricing_csv)):

        pricing = _load_pricing()

        # Check that pricing data was loaded correctly
        assert 'gpt-4o-mini-2024-07-18' in pricing
        assert pricing['gpt-4o-mini-2024-07-18']['input'] == 0.15
        assert pricing['gpt-4o-mini-2024-07-18']['output'] == 0.60
        assert 'gpt-4o-2024-08-06' in pricing
        assert pricing['gpt-4o-2024-08-06']['input'] == 1.00
        assert pricing['gpt-4o-2024-08-06']['output'] == 3.00


def test_load_pricing_file_not_found():
    """Test error handling when pricing file is not found."""
    with patch('os.path.exists', return_value=False):
        with pytest.raises(FileNotFoundError):
            _load_pricing()


def test_load_log_existing(mock_token_log):
    """Test loading an existing token log."""
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data=json.dumps(mock_token_log))):

        log = _load_log()

        # Should have loaded the mock data
        assert len(log) == 3
        assert log[0]['date'] == "2025-05-10"
        assert log[0]['tokens_submitted'] == 5000


def test_load_log_not_exists():
    """Test loading a non-existent token log."""
    with patch('os.path.exists', return_value=False):
        log = _load_log()
        # Should return an empty list
        assert log == []


def test_load_log_invalid_json():
    """Test loading an invalid JSON token log."""
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data="invalid json")):

        log = _load_log()
        # Should return an empty list on error
        assert log == []


def test_save_log(mock_token_log):
    """Test saving token log."""
    with patch('os.makedirs') as mock_makedirs, \
         patch('builtins.open', mock_open()) as mock_file:

        _save_log(mock_token_log)

        # Should have ensured directory exists
        mock_makedirs.assert_called_once()

        # Should have written the JSON data
        handle = mock_file()
        assert handle.write.call_count > 0  # Multiple writes for formatting

        # Join all the write calls to get the complete content
        write_calls = [args[0] for args, _ in handle.write.call_args_list]
        written_content = ''.join(write_calls)

        # Check the written content is valid JSON and matches our mock
        assert json.loads(written_content) == mock_token_log


def test_update_token_log_existing_entry(mock_token_log):
    """Test updating an existing entry in the token log."""
    with patch('src.token_tracker._load_log', return_value=mock_token_log.copy()), \
         patch('src.token_tracker._save_log') as mock_save:

        # Update existing entry
        update_token_log("sk-1234567890abcdefghijklmn",
                         5000,
                         date_str="2025-05-11")

        # Should have added tokens to the existing entry
        updated_log = mock_save.call_args[0][0]
        matching_entry = next(
            entry for entry in updated_log if entry['date'] == "2025-05-11"
            and entry['api_key_prefix'] == "sk-1234567**********")

        # The entry should have the tokens added to it
        assert matching_entry['tokens_submitted'] == 5000 + 10000


def test_update_token_log_new_entry(mock_token_log):
    """Test adding a new entry to the token log."""
    # Create a mock datetime module
    mock_dt = MagicMock()
    mock_dt.now.return_value = MagicMock()
    mock_dt.now.return_value.strftime.return_value = "2025-05-12"

    with patch('src.token_tracker._load_log', return_value=mock_token_log), \
         patch('src.token_tracker._save_log') as mock_save, \
         patch('src.token_tracker.datetime', mock_dt):

        # Add new entry for today
        update_token_log("sk-newkeyprefix", 2500)

        # Should have added a new entry
        updated_log = mock_save.call_args[0][0]
        assert len(updated_log) == 4

        # Check the new entry
        new_entry = updated_log[-1]
        assert new_entry['date'] == "2025-05-12"
        assert new_entry['api_key_prefix'] == "sk-newkeyp**********"
        assert new_entry['tokens_submitted'] == 2500


def test_log_token_usage_event(mock_pricing_csv):
    """Test logging a token usage event."""
    # Create a mock datetime module
    mock_dt = MagicMock()
    mock_dt.now.return_value = MagicMock()
    mock_dt.now.return_value.isoformat.return_value = "2025-05-12T10:30:00"

    with patch('os.path.exists', return_value=True), \
         patch('src.token_tracker._load_pricing', return_value={
             'gpt-4o-mini-2024-07-18': {'input': 0.15, 'output': 0.60}
         }), \
         patch('os.makedirs') as mock_makedirs, \
         patch('builtins.open', mock_open()) as mock_file, \
         patch('src.token_tracker.datetime', mock_dt):

        # Log an event
        log_token_usage_event(api_key="sk-1234567890abcdefghijklmn",
                              model="gpt-4o-mini-2024-07-18",
                              input_tokens=1000,
                              output_tokens=500)

        # Should have ensured directory exists
        mock_makedirs.assert_called_once()

        # Should have written the event
        handle = mock_file()
        handle.write.assert_called_once()

        # Check the written content
        written_content = handle.write.call_args[0][0]
        event = json.loads(written_content.strip())

        assert event['timestamp'] == "2025-05-12T10:30:00"
        assert event['api_key_prefix'] == "sk-1234567**********"
        assert event['model'] == "gpt-4o-mini-2024-07-18"
        assert event['input_tokens'] == 1000
        assert event['output_tokens'] == 500
        assert event['total_tokens'] == 1500
        # Cost calculation: (1000 * 0.15 + 500 * 0.60) / 1_000_000 = 0.00045
        assert event['cost'] == 0.00045


def test_log_token_usage_event_unknown_model():
    """Test logging an event with an unknown model."""
    with patch('src.token_tracker._load_pricing', return_value={}), \
         patch('os.makedirs'), \
         patch('builtins.open', mock_open()) as mock_file:

        # Log an event with an unknown model
        log_token_usage_event(api_key="sk-1234567890abcdefghijklmn",
                              model="unknown-model",
                              input_tokens=1000,
                              output_tokens=500,
                              timestamp="2025-05-12T10:30:00")

        # Should still write the event with zero cost
        handle = mock_file()
        written_content = handle.write.call_args[0][0]
        event = json.loads(written_content.strip())

        assert event['model'] == "unknown-model"
        assert event['cost'] == 0.0


def test_load_token_usage_events(mock_token_events):
    """Test loading token usage events."""
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data='\n'.join(mock_token_events))):

        events = load_token_usage_events()

        # Should have loaded all events
        assert len(events) == 3

        # Check first event
        assert events[0]['timestamp'] == "2025-05-10T10:00:00"
        assert events[0]['model'] == "gpt-4o-mini-2024-07-18"
        assert events[0]['input_tokens'] == 1000

        # Check another event
        assert events[2]['api_key_prefix'] == "sk-abcdefghij**********"
        assert events[2]['total_tokens'] == 4500


def test_get_token_usage_summary(mock_token_events):
    """Test aggregating token usage summary."""
    with patch('src.token_tracker.load_token_usage_events',
               return_value=[json.loads(e) for e in mock_token_events]):

        # Test grouping by day
        summary = get_token_usage_summary(group_by='day')

        assert summary['total_tokens'] == 9000  # 1500 + 3000 + 4500
        assert round(summary['total_cost'],
                     6) == 0.00680  # 0.00045 + 0.005 + 0.00135

        # Check breakdown by day
        assert len(summary['breakdown']) == 2  # 2 unique days
        assert "2025-05-10" in summary['breakdown']
        assert "2025-05-11" in summary['breakdown']
        assert summary['breakdown']["2025-05-10"]['tokens'] == 1500
        assert summary['breakdown']["2025-05-11"][
            'tokens'] == 7500  # 3000 + 4500

        # Test grouping by model
        model_summary = get_token_usage_summary(group_by='model')

        assert len(model_summary['breakdown']) == 2  # 2 unique models
        assert "gpt-4o-mini-2024-07-18" in model_summary['breakdown']
        assert "gpt-4o-2024-08-06" in model_summary['breakdown']
        assert model_summary['breakdown']["gpt-4o-mini-2024-07-18"][
            'tokens'] == 6000  # 1500 + 4500

        # Test date filtering
        filtered_summary = get_token_usage_summary(start_date="2025-05-11",
                                                   end_date="2025-05-11",
                                                   group_by='all')

        assert filtered_summary[
            'total_tokens'] == 7500  # Only events from 05-11
        assert round(filtered_summary['total_cost'],
                     6) == 0.00635  # 0.005 + 0.00135
        assert len(filtered_summary['breakdown']) == 1  # Just 'all'
        assert filtered_summary['breakdown']['all']['count'] == 2  # 2 events


def test_get_total_cost():
    """Test getting total cost for a date range."""
    with patch('src.token_tracker.get_token_usage_summary',
               return_value={'total_cost': 0.00680}):

        cost = get_total_cost(start_date="2025-05-10", end_date="2025-05-11")
        assert cost == 0.00680


def test_get_token_usage_for_day():
    """Test getting token usage for a specific day and API key."""
    # Create a simplified mock with the exact format we need
    mock_data = [{
        "date": "2025-05-11",
        "api_key_prefix": "sk-1234567**********",
        "tokens_submitted": 10000
    }, {
        "date": "2025-05-10",
        "api_key_prefix": "sk-1234567**********",
        "tokens_submitted": 5000
    }]

    with patch('src.token_tracker._load_log', return_value=mock_data):
        # Test with a key that matches the prefix in the mock data
        usage = get_token_usage_for_day("sk-1234567890abcdefghijklmn",
                                        date_str="2025-05-11")
        assert usage == 10000

        # Test with a different date
        usage = get_token_usage_for_day("sk-1234567890abcdefghijklmn",
                                        date_str="2025-05-10")
        assert usage == 5000

        # Test with a date that doesn't exist in the log
        usage = get_token_usage_for_day("sk-1234567890abcdefghijklmn",
                                        date_str="2025-05-12")
        assert usage == 0
