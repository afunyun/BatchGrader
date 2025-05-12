import pytest
from unittest.mock import MagicMock

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

    monkeypatch.setattr("src.llm_client.OpenAI", MockOpenAI)
