import pandas as pd
import pytest

from batchgrader.exceptions import BatchGraderFileNotFoundError, FileFormatError, TokenLimitError
from batchgrader.file_processor import process_file_common


class DummyClient:
    """A dummy LLM client for monkeypatching."""

    def __init__(self, config: dict = None):
        self.api_key = "dummy-key"
        self.config = config or {}

    def run_batch_job(
        self,
        input_df: pd.DataFrame,
        system_prompt: str,
        response_field_name: str,
        base_filename_for_tagging: str,
    ) -> pd.DataFrame:
        result_df = input_df.copy()
        result_df[response_field_name] = f"Processed with prompt: {system_prompt}"
        return result_df


@pytest.mark.parametrize(
    "input_file_content, expected_exception",
    [
        ("", FileFormatError),
        ("a,b\n", FileFormatError),
        ("a,b\n1,2", TokenLimitError),
    ],
)
def test_process_file_common(tmp_path, monkeypatch, input_file_content, expected_exception):
    """Test that exceptions are raised for invalid or empty files, and for token limit exceedance."""
    input_file = tmp_path / "data.csv"
    input_file.write_text(input_file_content)
    monkeypatch.setattr("batchgrader.file_processor.load_data", lambda path: pd.DataFrame())
    if expected_exception == TokenLimitError:
        monkeypatch.setattr(
            "batchgrader.file_processor.check_token_limits",
            lambda *a, **k: (False, {"total": 100}),
        )
        monkeypatch.setattr("batchgrader.file_processor.LLMClient", DummyClient)
        monkeypatch.setattr("batchgrader.file_processor.update_token_log", lambda *a, **k: None)
    with pytest.raises(expected_exception):
        process_file_common(str(input_file), str(tmp_path), {}, "prompt", "resp", None, 10)


def test_process_file_common_success(tmp_path, monkeypatch):
    """Test successful processing and token logging."""
    input_file = tmp_path / "data.csv"
    input_file.write_text("a,b\n1,2")
    df = pd.DataFrame({"a": [1], "b": [2]})
    monkeypatch.setattr("batchgrader.file_processor.load_data", lambda path: df)
    monkeypatch.setattr(
        "batchgrader.file_processor.check_token_limits",
        lambda *a, **k: (True, {"total": 10, "average": 10, "max": 10}),
    )
    monkeypatch.setattr("batchgrader.file_processor.LLMClient", DummyClient)

    def mock_update_token_log(api_key, tokens_submitted, tokens_used_for_error=False, log_path=None):
        mock_update_token_log.calls.append(
            {
                "api_key": api_key,
                "tokens_submitted": tokens_submitted,
                "tokens_used_for_error": tokens_used_for_error,
                "log_path": log_path,
            }
        )

    mock_update_token_log.calls = []
    monkeypatch.setattr("batchgrader.file_processor.update_token_log", mock_update_token_log)

    success, result = process_file_common(str(input_file), str(tmp_path), {}, "prompt", "resp", None, 10)

    assert success is True
    assert isinstance(result, pd.DataFrame)
    assert "resp" in result.columns
    assert mock_update_token_log.calls
    assert mock_update_token_log.calls[0]["tokens_submitted"] > 0

