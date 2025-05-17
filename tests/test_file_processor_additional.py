import logging
import os
from pathlib import Path
import pytest
import pandas as pd

import batchgrader.file_processor as fp

# Tests for prepare_output_path


def test_prepare_output_path_forced_results(tmp_path):
    config = {"force_chunk_count": 3}
    output_dir = tmp_path / "out"
    os.makedirs(output_dir, exist_ok=True)
    result = fp.prepare_output_path("input/testfile.txt", str(output_dir),
                                    config)
    assert result.endswith("testfile_forced_results.txt")


def test_prepare_output_path_permission_error(monkeypatch):
    monkeypatch.setattr(
        Path,
        "mkdir",
        lambda *args, **kwargs:
        (_ for _ in ()).throw(PermissionError("denied")),
    )
    with pytest.raises(fp.FilePermissionError) as exc:
        fp.prepare_output_path("input/file.txt", "some_dir", {})
    assert "Permission denied" in str(exc.value)


# Tests for calculate_and_log_token_usage


@pytest.fixture
def df():
    return pd.DataFrame({'a': [1], 'input_tokens': [10], 'output_tokens': [5]})

def test_calculate_and_log_token_usage_minimal(monkeypatch, caplog):
    monkeypatch.setattr(fp.CostEstimator, "estimate_cost",
                        lambda model, i, o: 1.23)
    calls = []
    monkeypatch.setattr(fp, "log_token_usage_event",
                        lambda **kwargs: calls.append(kwargs))
    df = pd.DataFrame({'a': [1], 'input_tokens': [10], 'output_tokens': [5]})
    fp.calculate_and_log_token_usage(df, "prompt", "resp", None, "my-model",
                                     "my-key")
    assert len(calls) == 1
    assert calls[0]["model"] == "my-model"
    assert any("Estimated LLM cost" in rec.message for rec in caplog.records)


def test_calculate_and_log_token_usage_logging_error(monkeypatch, caplog):
    monkeypatch.setattr(fp.CostEstimator, "estimate_cost",
                        lambda model, i, o: 0.0)

    def _raise(*args, **kwargs):
        raise logging.LoggingError("log error")

    monkeypatch.setattr(fp, "log_token_usage_event", _raise)
    caplog.set_level("ERROR", logger=fp.logger.name)
    fp.calculate_and_log_token_usage(df, "prompt", "resp", None, "model",
                                     "key")
    assert any("[Token Logging Error]" in rec.message
               for rec in caplog.records)
