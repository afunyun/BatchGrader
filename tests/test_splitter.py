"""
Test suite for input_splitter.py chunking logic and config merging.
Covers:
- force_chunk_count splitting
- token-based splitting
- error/warning for over-token chunks
- deep config merge
- existence check for examples file
- pricing.csv error handling

Test data is loaded from tests/input/ and temp files are written to tests/output/.
"""
import os
import sys
import shutil
import tempfile
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.input_splitter import (split_file_by_token_limit, InputSplitterError,
                             MissingArgumentError, FileNotFoundError,
                             UnsupportedFileTypeError, OutputDirectoryError)
from src.utils import deep_merge_dicts
from src.constants import DEFAULT_PRICING_CSV_PATH
from src.token_tracker import _load_pricing

TEST_INPUT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), 'input'))
TEST_OUTPUT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), 'output'))


def dummy_token_counter(row):
    return len(str(row.get('text', ''))) if hasattr(row, 'get') else len(
        str(row))


@pytest.fixture(scope='function')
def cleanup_output():
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    yield
    shutil.rmtree(TEST_OUTPUT_DIR)
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)


def test_force_chunk_count_splits_evenly(cleanup_output):
    df = pd.DataFrame({'text': [f"row {i}" for i in range(10)]})
    tmp = os.path.join(TEST_OUTPUT_DIR, 'force_chunk.csv')
    df.to_csv(tmp, index=False)
    
    # Create a mock logger that won't cause TypeError with level comparison
    mock_logger = MagicMock()
    
    with patch('src.input_splitter.logger', mock_logger):
        files, tokens = split_file_by_token_limit(
            tmp,
            token_limit=100,
            count_tokens_fn=dummy_token_counter,
            force_chunk_count=3,
            output_dir=TEST_OUTPUT_DIR)
    assert len(files) == 3
    total_rows = sum(pd.read_csv(f).shape[0] for f in files)
    assert total_rows == 10


def test_token_chunking_respects_limit(cleanup_output):
    import pandas as pd
    import os
    from unittest.mock import MagicMock, patch
    from src.input_splitter import split_file_by_token_limit
    from tests.test_helpers import dummy_token_counter, TEST_OUTPUT_DIR

    df = pd.DataFrame({'text': ['a' * 10] * 12})
    tmp = os.path.join(TEST_OUTPUT_DIR, 'tokchunk.csv')
    df.to_csv(tmp, index=False)
    mock_logger = MagicMock()
    with patch('src.input_splitter.logger', mock_logger):
        files, tokens = split_file_by_token_limit(
            tmp,
            token_limit=30,
            count_tokens_fn=dummy_token_counter,
            output_dir=TEST_OUTPUT_DIR)
    assert all(token_count <= 30 for token_count in tokens)
    all_rows = sum(pd.read_csv(f).shape[0] for f in files)
    assert all_rows == 12


def test_force_chunk_count_warns_over_token(monkeypatch, cleanup_output):
    df = pd.DataFrame({'text': ['x' * 50] * 5})
    tmp = os.path.join(TEST_OUTPUT_DIR, 'overchunk.csv')
    df.to_csv(tmp, index=False)
    warnings = []

    def fake_logger():

        class L:

            def warning(self, msg):
                warnings.append(msg)

            def event(self, msg):
                pass

            def debug(self, msg):
                pass

            def info(self, msg):
                pass

            def error(self, msg):
                pass

        return L()

    files, tokens = split_file_by_token_limit(
        tmp,
        token_limit=100,
        count_tokens_fn=dummy_token_counter,
        force_chunk_count=2,
        output_dir=TEST_OUTPUT_DIR,
        logger_override=fake_logger())
    assert any('exceeds token limit' in w for w in warnings)


def test_deep_merge_dicts():
    a = {'a': 1, 'b': {'c': 2, 'd': 3}}
    b = {'b': {'d': 4, 'e': 5}, 'f': 6}
    merged = deep_merge_dicts(a, b)
    assert merged['b']['c'] == 2
    assert merged['b']['d'] == 4
    assert merged['b']['e'] == 5
    assert merged['f'] == 6


def test_examples_file_check(tmp_path):
    from config_loader import is_examples_file_default, DEFAULT_EXAMPLES_TEXT
    p = tmp_path / 'examples.txt'
    p.write_text(DEFAULT_EXAMPLES_TEXT)
    assert is_examples_file_default(str(p))
    p.write_text('not default')
    assert not is_examples_file_default(str(p))


def test_pricing_csv_error(monkeypatch):
    monkeypatch.setattr(Path, 'exists', lambda self: False)
    with pytest.raises(FileNotFoundError):
        _load_pricing(DEFAULT_PRICING_CSV_PATH)
