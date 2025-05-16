"""
Unit tests for the token_tracker module.
"""

import json
import os
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock, call

from token_tracker import (_get_api_key_prefix, _load_pricing, _load_log,
                           _save_log, update_token_log, log_token_usage_event,
                           load_token_usage_events, get_token_usage_summary,
                           get_total_cost, get_token_usage_for_day)


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
    # Explicitly create a list of separate strings
    event1 = json.dumps({
        "timestamp": "2025-05-10T10:00:00",
        "api_key_prefix": "sk-1234567890**********",
        "model": "gpt-4o-mini-2024-07-18",
        "input_tokens": 1000,
        "output_tokens": 500,
        "total_tokens": 1500,
        "cost": 0.00045
    })
    event2 = json.dumps({
        "timestamp": "2025-05-11T11:00:00",
        "api_key_prefix": "sk-1234567890**********",
        "model": "gpt-4o-2024-08-06",
        "input_tokens": 2000,
        "output_tokens": 1000,
        "total_tokens": 3000,
        "cost": 0.005
    })
    event3 = json.dumps({
        "timestamp": "2025-05-11T12:00:00",
        "api_key_prefix": "sk-abcdefghij**********",
        "model": "gpt-4o-mini-2024-07-18",
        "input_tokens": 3000,
        "output_tokens": 1500,
        "total_tokens": 4500,
        "cost": 0.00135
    })
    return [event1, event2, event3]


@pytest.fixture
def temp_log_path(tmp_path: Path) -> Path:
    return tmp_path / "token_usage_log.json"


@pytest.fixture
def temp_event_log_path(tmp_path: Path) -> Path:
    return tmp_path / "token_usage_events.jsonl"


@pytest.fixture
def temp_pricing_csv_path(tmp_path: Path, mock_pricing_csv: str) -> Path:
    pricing_file = tmp_path / "pricing.csv"
    pricing_file.write_text(mock_pricing_csv)
    return pricing_file


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


def test_load_pricing(temp_pricing_csv_path: Path):
    """Test loading pricing data from CSV."""
    # temp_pricing_csv_path fixture already creates the file with content
    pricing = _load_pricing(temp_pricing_csv_path)

    # Check that pricing data was loaded correctly
    assert 'gpt-4o-mini-2024-07-18' in pricing
    assert pricing['gpt-4o-mini-2024-07-18']['input'] == 0.15
    assert pricing['gpt-4o-mini-2024-07-18']['output'] == 0.60
    assert 'gpt-4o-2024-08-06' in pricing
    assert pricing['gpt-4o-2024-08-06']['input'] == 1.00
    assert pricing['gpt-4o-2024-08-06']['output'] == 3.00


def test_load_pricing_file_not_found(tmp_path: Path):
    """Test error handling when pricing file is not found."""
    non_existent_path = tmp_path / "non_existent_pricing.csv"
    with pytest.raises(FileNotFoundError):
        _load_pricing(non_existent_path)


def test_load_log_existing(mock_token_log, temp_log_path: Path):
    """Test loading an existing token log."""
    temp_log_path.write_text(json.dumps(mock_token_log))
    log = _load_log(temp_log_path)

    # Should have loaded the mock data
    assert len(log) == 3
    assert log[0]['date'] == "2025-05-10"
    assert log[0]['tokens_submitted'] == 5000


def test_load_log_not_exists(temp_log_path: Path):
    """Test loading a non-existent token log."""
    log = _load_log(temp_log_path)
    # Should return an empty list
    assert log == []


def test_load_log_invalid_json(temp_log_path: Path):
    """Test loading an invalid JSON token log."""
    temp_log_path.write_text("invalid json")
    log = _load_log(temp_log_path)
    # Should return an empty list on error
    assert log == []


def test_save_log(mock_token_log, temp_log_path: Path):
    """Test saving token log."""
    _save_log(mock_token_log, temp_log_path)

    # Should have ensured directory exists (implicitly by writing file)
    assert temp_log_path.exists()

    # Check the written content is valid JSON and matches our mock
    assert json.loads(temp_log_path.read_text()) == mock_token_log


def test_update_token_log_existing_entry(mock_token_log, temp_log_path: Path):
    """Test updating an existing entry in the token log."""
    # Prepare initial log file
    temp_log_path.write_text(json.dumps(mock_token_log.copy()))

    # Update existing entry
    update_token_log("sk-1234567890abcdefghijklmn",
                     5000,
                     log_path=temp_log_path,
                     date_str="2025-05-11")

    # Should have added tokens to the existing entry
    updated_log_content = json.loads(temp_log_path.read_text())
    matching_entry = next(
        entry for entry in updated_log_content if entry['date'] == "2025-05-11"
        and entry['api_key_prefix'] == "sk-1234567**********")

    # The entry should have the tokens added to it
    assert matching_entry['tokens_submitted'] == 5000 + 10000


def test_update_token_log_new_entry(mock_token_log, temp_log_path: Path):
    """Test adding a new entry to the token log."""
    # Prepare initial log file
    temp_log_path.write_text(json.dumps(mock_token_log))

    # Create a fixed datetime for testing
    fixed_date = "2025-05-16"
    mock_dt = MagicMock()
    mock_dt.now.return_value.strftime.return_value = fixed_date

    with patch('token_tracker.datetime', mock_dt):
        update_token_log("sk-newkeyprefix", 2500, log_path=temp_log_path)

        updated_log_content = json.loads(temp_log_path.read_text())
        assert len(updated_log_content) == 4

        new_entry = updated_log_content[-1]
        assert new_entry['date'] == fixed_date
        assert new_entry['api_key_prefix'] == "sk-newkeyp**********"
        assert new_entry['tokens_submitted'] == 2500


def test_log_token_usage_event(temp_event_log_path: Path,
                               temp_pricing_csv_path: Path):
    """Test logging a token usage event."""
    # Create a fixed datetime for testing
    fixed_timestamp = "2025-05-16T02:01:23.323343"
    mock_dt = MagicMock()
    mock_dt.now.return_value.isoformat.return_value = fixed_timestamp

    with patch('token_tracker.datetime', mock_dt):
        log_token_usage_event(api_key="sk-1234567890abcdefghijklmn",
                              model="gpt-4o-mini-2024-07-18",
                              input_tokens=1000,
                              output_tokens=500,
                              event_log_path=temp_event_log_path,
                              pricing_csv_path=temp_pricing_csv_path)

        assert temp_event_log_path.exists()
        written_content = temp_event_log_path.read_text().strip()
        event = json.loads(written_content)

        assert event['timestamp'] == fixed_timestamp
        assert event['api_key_prefix'] == "sk-1234567**********"
        assert event['model'] == "gpt-4o-mini-2024-07-18"
        assert event['input_tokens'] == 1000
        assert event['output_tokens'] == 500
        assert event['total_tokens'] == 1500
        # Cost calculation: (1000 * 0.15 + 500 * 0.60) / 1_000_000 = 0.00045
        assert event['cost'] == 0.00045


def test_log_token_usage_event_unknown_model(temp_event_log_path: Path,
                                             temp_pricing_csv_path: Path):
    """Test logging an event with an unknown model."""
    # Ensure pricing CSV doesn't contain 'unknown-model'
    # temp_pricing_csv_path already set up by fixture

    log_token_usage_event(api_key="sk-1234567890abcdefghijklmn",
                          model="unknown-model",
                          input_tokens=1000,
                          output_tokens=500,
                          event_log_path=temp_event_log_path,
                          pricing_csv_path=temp_pricing_csv_path,
                          timestamp="2025-05-12T10:30:00")

    assert temp_event_log_path.exists()
    written_content = temp_event_log_path.read_text().strip()
    event = json.loads(written_content)

    assert event['model'] == "unknown-model"
    assert event['cost'] == 0.0


def test_load_token_usage_events(mock_token_events, temp_event_log_path: Path):
    """Test loading token usage events."""
    # Diagnostic: write line by line
    with open(temp_event_log_path, 'w', encoding='utf-8') as f:
        for event_str in mock_token_events:
            f.write(event_str + '\n')

    events = load_token_usage_events(temp_event_log_path)
    assert len(events) == 3
    assert events[0]['input_tokens'] == 1000
    assert events[1]['model'] == "gpt-4o-2024-08-06"
    assert events[2]['cost'] == 0.00135


def test_get_token_usage_summary(mock_token_events, temp_event_log_path: Path):
    """Test aggregating token usage summary."""
    # Diagnostic: write line by line
    with open(temp_event_log_path, 'w', encoding='utf-8') as f:
        for event_str in mock_token_events:
            f.write(event_str + '\n')

    # Test grouping by day
    summary_day = get_token_usage_summary(event_log_path=temp_event_log_path,
                                          group_by='day')
    assert summary_day['total_tokens'] == 9000  # 1500 + 3000 + 4500
    assert round(summary_day['total_cost'],
                 6) == 0.00680  # 0.00045 + 0.005 + 0.00135

    # Check breakdown by day
    assert len(summary_day['breakdown']) == 2  # 2 unique days
    assert "2025-05-10" in summary_day['breakdown']
    assert "2025-05-11" in summary_day['breakdown']
    assert summary_day['breakdown']["2025-05-10"]['tokens'] == 1500
    assert summary_day['breakdown']["2025-05-11"][
        'tokens'] == 7500  # 3000 + 4500

    # Test grouping by model
    summary_model = get_token_usage_summary(event_log_path=temp_event_log_path,
                                            group_by='model')
    assert len(summary_model['breakdown']) == 2  # 2 unique models
    assert "gpt-4o-mini-2024-07-18" in summary_model['breakdown']
    assert "gpt-4o-2024-08-06" in summary_model['breakdown']
    assert summary_model['breakdown']["gpt-4o-mini-2024-07-18"][
        'tokens'] == 6000  # 1500 + 4500

    # Test date filtering
    filtered_summary = get_token_usage_summary(
        event_log_path=temp_event_log_path,
        start_date="2025-05-11",
        end_date="2025-05-11",
        group_by='all')

    assert filtered_summary['total_tokens'] == 7500  # Only events from 05-11
    assert round(filtered_summary['total_cost'],
                 6) == 0.00635  # 0.005 + 0.00135
    assert len(filtered_summary['breakdown']) == 1  # Just 'all'
    assert filtered_summary['breakdown']['all']['count'] == 2  # 2 events


def test_get_total_cost(temp_event_log_path: Path):
    """Test getting total cost for a date range."""
    dummy_events = [
        json.dumps({
            "timestamp": "2025-05-10T10:00:00",
            "model": "m1",
            "input_tokens": 10,
            "output_tokens": 10,
            "total_tokens": 20,
            "cost": 0.1
        }),
        json.dumps({
            "timestamp": "2025-05-11T10:00:00",
            "model": "m1",
            "input_tokens": 10,
            "output_tokens": 10,
            "total_tokens": 20,
            "cost": 0.2
        })
    ]
    # Diagnostic: write line by line
    with open(temp_event_log_path, 'w', encoding='utf-8') as f:
        for event_str in dummy_events:
            f.write(event_str + '\n')

    cost = get_total_cost(event_log_path=temp_event_log_path,
                          start_date="2025-05-10",
                          end_date="2025-05-11")
    assert cost == 0.3

    cost_one_day = get_total_cost(event_log_path=temp_event_log_path,
                                  start_date="2025-05-10",
                                  end_date="2025-05-10")
    assert cost_one_day == 0.1


def test_get_token_usage_for_day(temp_log_path: Path):
    """Test getting token usage for a specific day and API key."""
    mock_data = [{
        "date": "2025-05-11",
        "api_key_prefix": "sk-1234567**********",
        "tokens_submitted": 10000
    }, {
        "date": "2025-05-10",
        "api_key_prefix": "sk-1234567**********",
        "tokens_submitted": 5000
    }]
    temp_log_path.write_text(json.dumps(mock_data))

    # Test with a key that matches the prefix in the mock data
    usage = get_token_usage_for_day("sk-1234567890abcdefghijklmn",
                                    log_path=temp_log_path,
                                    date_str="2025-05-11")
    assert usage == 10000

    # Test with a different date
    usage = get_token_usage_for_day("sk-1234567890abcdefghijklmn",
                                    log_path=temp_log_path,
                                    date_str="2025-05-10")
    assert usage == 5000

    # Test with a date that doesn't exist in the log
    usage = get_token_usage_for_day("sk-1234567890abcdefghijklmn",
                                    log_path=temp_log_path,
                                    date_str="2025-05-12")
    assert usage == 0

    # Test with a key that doesn't exist
    usage = get_token_usage_for_day("sk-unknown**********",
                                    log_path=temp_log_path,
                                    date_str="2025-05-11")
    assert usage == 0

    # Test with an empty log file
    empty_log_path = temp_log_path.parent / "empty_log.json"
    empty_log_path.write_text("[]")
    usage = get_token_usage_for_day("sk-1234567890abcdefghijklmn",
                                    log_path=empty_log_path,
                                    date_str="2025-05-11")
    assert usage == 0
