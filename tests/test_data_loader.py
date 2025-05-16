import os
import pytest
import pandas as pd
import json
import tempfile
from data_loader import load_data, save_data


@pytest.fixture
def sample_df():
    """Create a sample dataframe for testing."""
    return pd.DataFrame({
        'id': [1, 2, 3],
        'text': ['First row', 'Second row', 'Third row'],
        'value': [10.1, 20.2, 30.3]
    })


@pytest.fixture
def temp_csv_file(sample_df):
    """Create a temporary CSV file with sample data."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv",
                                     delete=False) as tmp:
        sample_df.to_csv(tmp.name, index=False)
        tmp_path = tmp.name

    yield tmp_path

    # Clean up
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)


@pytest.fixture
def temp_json_file(sample_df):
    """Create a temporary JSON file with sample data."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                     delete=False) as tmp:
        sample_df.to_json(tmp.name, orient='records')
        tmp_path = tmp.name

    yield tmp_path

    # Clean up
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)


@pytest.fixture
def temp_jsonl_file(sample_df):
    """Create a temporary JSONL file with sample data."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl",
                                     delete=False) as tmp:
        sample_df.to_json(tmp.name, orient='records', lines=True)
        tmp_path = tmp.name

    yield tmp_path

    # Clean up
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)


def test_load_csv_file(temp_csv_file):
    """Test loading data from a CSV file."""
    df = load_data(temp_csv_file)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert list(df.columns) == ['id', 'text', 'value']
    assert df['id'].tolist() == [1, 2, 3]
    assert df['text'].tolist() == ['First row', 'Second row', 'Third row']


def test_load_json_file(temp_json_file):
    """Test loading data from a JSON file."""
    df = load_data(temp_json_file)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert set(df.columns) == {'id', 'text', 'value'}
    assert df['id'].tolist() == [1, 2, 3]
    assert df['text'].tolist() == ['First row', 'Second row', 'Third row']


def test_load_jsonl_file(temp_jsonl_file):
    """Test loading data from a JSONL file."""
    df = load_data(temp_jsonl_file)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert set(df.columns) == {'id', 'text', 'value'}
    assert df['id'].tolist() == [1, 2, 3]
    assert df['text'].tolist() == ['First row', 'Second row', 'Third row']


def test_save_data_csv(sample_df, tmp_path):
    """Test saving data to a CSV file."""
    output_path = os.path.join(tmp_path, "output.csv")
    save_data(sample_df, output_path)

    # Check that file exists
    assert os.path.exists(output_path)

    # Load file and check contents
    saved_df = pd.read_csv(output_path)
    pd.testing.assert_frame_equal(saved_df, sample_df)


def test_save_data_json(sample_df, tmp_path):
    """Test saving data to a JSON file."""
    output_path = os.path.join(tmp_path, "output.json")
    save_data(sample_df, output_path)

    # Check that file exists
    assert os.path.exists(output_path)

    # Load file and check contents
    saved_df = pd.read_json(output_path)
    pd.testing.assert_frame_equal(saved_df, sample_df)


def test_save_data_jsonl(sample_df, tmp_path):
    """Test saving data to a JSONL file."""
    output_path = os.path.join(tmp_path, "output.jsonl")
    save_data(sample_df, output_path)

    # Check that file exists
    assert os.path.exists(output_path)

    # Load file and check contents
    saved_df = pd.read_json(output_path, lines=True)
    pd.testing.assert_frame_equal(saved_df, sample_df)


def test_unsupported_file_format_load():
    """Test that loading an unsupported file format raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported file format"):
        load_data("file.txt")


def test_unsupported_file_format_save(sample_df):
    """Test that saving to an unsupported file format raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported file format"):
        save_data(sample_df, "file.txt")


def test_load_nonexistent_file():
    """Test that loading a nonexistent file raises an appropriate error."""
    with pytest.raises(
            Exception
    ):  # Could be FileNotFoundError or other exceptions depending on pandas behavior
        load_data("nonexistent_file.csv")
