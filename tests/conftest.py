"""
Configuration file for pytest.
This file contains fixtures and configuration for the test suite.
"""
import os
import sys
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Mock the tiktoken module more comprehensively
mock_tiktoken = MagicMock()
mock_encoder = MagicMock()
mock_encoder.encode = MagicMock(return_value=[1, 2, 3])  # Return some token IDs
mock_encoder.decode = MagicMock(return_value="decoded text")
mock_tiktoken.get_encoding = MagicMock(return_value=mock_encoder)
sys.modules['tiktoken'] = mock_tiktoken

# Mock the openai module at the module level
sys.modules['openai'] = MagicMock()

# Ignore YAML files in tests/config from collection
collect_ignore_glob = ["config/*.yaml", "config/*.yml"]

# Configure pytest to ignore YAML files
# This hook might be redundant if collect_ignore_glob works effectively for all collectors.
# Keeping it simplified as a fallback.
def pytest_collect_file(file_path, parent):
    """Skip YAML files if they somehow bypass other ignore mechanisms."""
    if file_path.suffix.lower() in [".yaml", ".yml"]:
        return None
    # For other files, let pytest's default collection proceed.

@pytest.fixture
def basic_config():
    """Provide a basic configuration for tests."""
    return {
        'global_token_limit': 2000000,
        'input_splitter_options': {},
        'llm_client_options': {'api_key': 'TEST_KEY'},
        'openai_model_name': 'gpt-3.5-turbo',
        'examples_dir': 'tests/input/examples',
        'system_prompt_template': 'tests/config/test_system_prompt.txt'
    }

@pytest.fixture(autouse=True)
def mock_openai_batch(monkeypatch):
    """
    Automatically mock OpenAI API client for all tests.
    Simulates real conditions: returns errors for malformed requests, otherwise returns success.
    """
    class MockBatch:
        def __init__(self, id="mock_batch_id", status="completed"):
            self.id = id
            self.status = status

    class MockBatches:
        def create(self, *args, **kwargs):
            return MockBatch()
            
        def retrieve(self, batch_id):
            return MockBatch(id=batch_id)
            
        async def create_async(self, *args, **kwargs):
            return self.create(*args, **kwargs)
            
        async def retrieve_async(self, batch_id):
            return self.retrieve(batch_id)

    class MockFiles:
        def create(self, *args, **kwargs):
            return MagicMock(id="mock_file_id")
        def content(self, file_id):
            if file_id == "malformed_file_id":
                raise Exception("Malformed file ID")
            valid_jsonl = (
                '{"custom_id": "1", "response": {"status_code": 200, "body": {"choices": [{"message": {"content": "MOCKED_RESPONSE"}}]}}}\n'
                '{"custom_id": "2", "error": {"code": "invalid_request", "message": "Malformed request"}}\n'
            )
            return MagicMock(text=valid_jsonl)

    class MockOpenAI:
        def __init__(self, *args, **kwargs):
            self.batches = MockBatches()
            self.files = MockFiles()

    monkeypatch.setattr("llm_client.OpenAI", MockOpenAI)
