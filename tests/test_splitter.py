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
# Add project root to sys.path, so 'src' is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.input_splitter import split_file_by_token_limit
from src.utils import deep_merge_dicts

TEST_INPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'input'))
TEST_OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'output'))

# --- Utility: Dummy token counter ---
def dummy_token_counter(row):
    # Assume each row is a dict with a 'text' field
    return len(str(row.get('text', ''))) if hasattr(row, 'get') else len(str(row))

@pytest.fixture(scope='function')
def cleanup_output():
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    yield
    shutil.rmtree(TEST_OUTPUT_DIR)
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)

# --- Test: force_chunk_count splitting ---
def test_force_chunk_count_splits_evenly(cleanup_output):
    df = pd.DataFrame({'text': [f"row {i}" for i in range(10)]})
    tmp = os.path.join(TEST_OUTPUT_DIR, 'force_chunk.csv')
    df.to_csv(tmp, index=False)
    files, tokens = split_file_by_token_limit(
        tmp, token_limit=100, count_tokens_fn=dummy_token_counter,
        force_chunk_count=3, output_dir=TEST_OUTPUT_DIR)
    assert len(files) == 3
    total_rows = sum(pd.read_csv(f).shape[0] for f in files)
    assert total_rows == 10

# --- Test: token-based splitting ---
def test_token_chunking_respects_limit(cleanup_output):
    df = pd.DataFrame({'text': ['a'*10]*12})
    tmp = os.path.join(TEST_OUTPUT_DIR, 'tokchunk.csv')
    df.to_csv(tmp, index=False)
    files, tokens = split_file_by_token_limit(
        tmp, token_limit=30, count_tokens_fn=dummy_token_counter,
        output_dir=TEST_OUTPUT_DIR)
    # Each chunk should have <= 30 tokens
    for t in tokens:
        assert t <= 30
    all_rows = sum(pd.read_csv(f).shape[0] for f in files)
    assert all_rows == 12

# --- Test: warning for over-token chunk in forced mode ---
def test_force_chunk_count_warns_over_token(monkeypatch, cleanup_output):
    df = pd.DataFrame({'text': ['x'*50]*5})
    tmp = os.path.join(TEST_OUTPUT_DIR, 'overchunk.csv')
    df.to_csv(tmp, index=False)
    warnings = []
    def fake_logger():
        class L:
            def warning(self, msg):
                warnings.append(msg)
            def event(self, msg):
                pass
        return L()
    files, tokens = split_file_by_token_limit(
        tmp, token_limit=100, count_tokens_fn=dummy_token_counter,
        force_chunk_count=2, output_dir=TEST_OUTPUT_DIR, logger=fake_logger())
    assert any('exceeds token limit' in w for w in warnings)

# --- Test: deep config merge ---
def test_deep_merge_dicts():
    a = {'a': 1, 'b': {'c': 2, 'd': 3}}
    b = {'b': {'d': 4, 'e': 5}, 'f': 6}
    merged = deep_merge_dicts(a, b)
    assert merged['b']['c'] == 2
    assert merged['b']['d'] == 4
    assert merged['b']['e'] == 5
    assert merged['f'] == 6

# --- Test: examples file existence check ---
def test_examples_file_check(tmp_path):
    from src.config_loader import is_examples_file_default, DEFAULT_EXAMPLES_TEXT
    # Should be True for default content
    p = tmp_path / 'examples.txt'
    p.write_text(DEFAULT_EXAMPLES_TEXT)
    assert is_examples_file_default(str(p))
    # Should be False for custom content
    p.write_text('not default')
    assert not is_examples_file_default(str(p))

# --- Test: pricing.csv error handling ---
def test_pricing_csv_error(monkeypatch):
    from src.token_tracker import _load_pricing, PRICING_CSV_PATH
    # Rename real pricing.csv
    if os.path.exists(PRICING_CSV_PATH):
        os.rename(PRICING_CSV_PATH, PRICING_CSV_PATH+'.bak')
    try:
        try:
            _ = _load_pricing()
            assert False, "Should raise FileNotFoundError"
        except FileNotFoundError:
            pass
    finally:
        if os.path.exists(PRICING_CSV_PATH+'.bak'):
            os.rename(PRICING_CSV_PATH+'.bak', PRICING_CSV_PATH)
