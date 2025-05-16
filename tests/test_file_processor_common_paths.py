import pytest
import pandas as pd
from pathlib import Path
import os

import src.file_processor as fp
from src.exceptions import FileNotFoundError as FGFileNotFoundError, FileFormatError, TokenLimitError

# Monkeypatch helpers
class DummyClient:
    def __init__(self):
        self.api_key = 'dummy-key'


def test_process_file_common_not_found(tmp_path):
    # Path does not exist
    p = tmp_path / "nofile.csv"
    with pytest.raises(FGFileNotFoundError):
        fp.process_file_common(str(p), str(tmp_path), {}, "prompt", "resp", None, 10)


def test_process_file_common_format_error(tmp_path, monkeypatch):
    # Create empty file
    p = tmp_path / "data.csv"
    p.write_text('')
    monkeypatch.setattr(fp, 'load_data', lambda path: None)
    with pytest.raises(FileFormatError):
        fp.process_file_common(str(p), str(tmp_path), {}, "prompt", "resp", None, 10)


def test_process_file_common_empty_df(tmp_path, monkeypatch):
    p = tmp_path / "data.csv"
    p.write_text('a,b\n')
    monkeypatch.setattr(fp, 'load_data', lambda path: pd.DataFrame())
    success, result = fp.process_file_common(str(p), str(tmp_path), {}, "prompt", "resp", None, 10)
    assert success is False and result is None


def test_process_file_common_token_limit_exceeded(tmp_path, monkeypatch):
    # Create dummy CSV path and DataFrame
    p = tmp_path / "data.csv"
    p.write_text('a,b\n1,2')
    df = pd.DataFrame({'a':[1],'b':[2]})
    # load_data returns df
    monkeypatch.setattr(fp, 'load_data', lambda path: df)
    # check_token_limits returns under_limit False
    monkeypatch.setattr(fp, 'check_token_limits', lambda *args, **kwargs: (False, {'total':100}))
    # patch LLMClient and update_token_log
    monkeypatch.setattr(fp, 'LLMClient', DummyClient)
    monkeypatch.setattr(fp, 'update_token_log', lambda *args, **kwargs: None)
    with pytest.raises(TokenLimitError):
        fp.process_file_common(str(p), str(tmp_path), {}, "prompt", "resp", None, 10)


def test_process_file_common_success(tmp_path, monkeypatch):
    p = tmp_path / "data.csv"
    p.write_text('a,b\n1,2')
    df = pd.DataFrame({'a':[1],'b':[2]})
    # load_data returns df
    monkeypatch.setattr(fp, 'load_data', lambda path: df)
    # check_token_limits returns under_limit True
    monkeypatch.setattr(fp, 'check_token_limits', lambda *args, **kwargs: (True, {'total':10}))
    # patch LLMClient and update_token_log
    monkeypatch.setattr(fp, 'LLMClient', DummyClient)
    calls = []
    monkeypatch.setattr(fp, 'update_token_log', lambda *args, **kwargs: calls.append((args,kwargs)))
    success, result = fp.process_file_common(str(p), str(tmp_path), {}, "prompt", "resp", None, 10)
    assert success is True and isinstance(result, pd.DataFrame)
    # ensure update_token_log was called for logging usage
    assert calls and calls[0][0][1] == 10
